/**
 * Tests for the DeerStop WebRTC signaling server.
 *
 * Uses Node.js built-in test runner (node:test) and the `ws` package.
 * The server starts on a random port once for all tests.
 *
 * Run: node --test node/signaling/server.test.js
 *   or: cd node/signaling && npm test
 */

"use strict";

const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const { WebSocket } = require("ws");

// ---------------------------------------------------------------------------
// Test environment — set before requiring server so constants are picked up
// ---------------------------------------------------------------------------

const TEST_TOKEN  = "test-token-123";
const TEST_SECRET = "test-secret-abc";
const TEST_TURN   = "turn:test.example.com:3478";

process.env.AUTH_TOKEN  = TEST_TOKEN;
process.env.TURN_SECRET = TEST_SECRET;
process.env.TURN_URL    = TEST_TURN;
process.env.PORT        = "0"; // random port

const { server } = require("./server");

let baseUrl;

before(async () => {
  await new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      baseUrl = `ws://127.0.0.1:${port}`;
      resolve();
    });
  });
});

after(async () => {
  await new Promise((resolve) => server.close(resolve));
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Create a WebSocket and immediately attach message/error/close listeners
 * BEFORE the socket opens, so we never miss an early message.
 * Returns a promise that resolves with { ws, firstMessage } once the first
 * message is received (or rejects on error/unexpected close).
 */
function connectAndWaitForFirstMessage(token = TEST_TOKEN) {
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const ws = new WebSocket(baseUrl, { headers });

  const firstMessage = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("waitForMessage timeout")), 5000);
    ws.once("message", (raw) => {
      clearTimeout(timer);
      try { resolve(JSON.parse(raw)); } catch (e) { reject(e); }
    });
    ws.once("error", (err) => { clearTimeout(timer); reject(err); });
  });

  return { ws, firstMessage };
}

/** Open a socket and wait for the first message (turn_credentials). */
async function connectAndGetTurnCredentials(token = TEST_TOKEN) {
  const { ws, firstMessage } = connectAndWaitForFirstMessage(token);
  const msg = await firstMessage;
  return { ws, msg };
}

/** Register as streamer or viewer; returns { ws } after registration. */
async function registerAs(role) {
  const { ws, firstMessage } = connectAndWaitForFirstMessage();
  await firstMessage; // consume turn_credentials
  ws.send(JSON.stringify({ type: "register", role }));
  return ws;
}

/** Wait for the next message on an already-open socket (with timeout). */
function nextMessage(ws, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("nextMessage timeout")), timeoutMs);
    ws.once("message", (raw) => {
      clearTimeout(timer);
      try { resolve(JSON.parse(raw)); } catch (e) { reject(e); }
    });
    ws.once("error", (err) => { clearTimeout(timer); reject(err); });
  });
}

/** Wait for the socket to close; resolves with { code }. */
function waitForClose(ws) {
  return new Promise((resolve) => {
    ws.once("close", (code) => resolve({ code }));
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test("rejects unauthenticated connection", async () => {
  const headers = {};
  const ws = new WebSocket(baseUrl, { headers });
  const { code } = await waitForClose(ws);
  assert.equal(code, 4401);
});

test("rejects wrong token", async () => {
  const ws = new WebSocket(baseUrl, { headers: { Authorization: "Bearer wrong" } });
  const { code } = await waitForClose(ws);
  assert.equal(code, 4401);
});

test("accepts authenticated connection", async () => {
  const { ws } = await connectAndGetTurnCredentials();
  ws.close();
});

test("sends turn_credentials immediately on connect", async () => {
  const { ws, msg } = await connectAndGetTurnCredentials();
  assert.equal(msg.type, "turn_credentials");
  assert.ok(msg.username, "username should be present");
  assert.ok(msg.credential, "credential should be present");
  assert.equal(msg.turn_url, TEST_TURN);
  ws.close();
});

test("turn credentials username encodes a future timestamp", async () => {
  const { ws, msg } = await connectAndGetTurnCredentials();
  const ts = parseInt(msg.username, 10);
  assert.ok(ts > Math.floor(Date.now() / 1000), "username should be a future unix timestamp");
  ws.close();
});

test("notifies streamer when viewer connects", async () => {
  const streamer = await registerAs("streamer");
  const viewer   = await registerAs("viewer");

  const notification = await nextMessage(streamer);
  assert.equal(notification.type, "viewer_connected");
  assert.ok(notification.viewer_id, "viewer_id should be present");

  streamer.close();
  viewer.close();
});

test("relays sdp_offer from streamer to viewer", async () => {
  const streamer = await registerAs("streamer");
  const viewer   = await registerAs("viewer");

  const { viewer_id } = await nextMessage(streamer);

  const offerSdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n";
  streamer.send(JSON.stringify({ type: "sdp_offer", viewer_id, sdp: offerSdp }));

  const received = await nextMessage(viewer);
  assert.equal(received.type, "sdp_offer");
  assert.equal(received.sdp, offerSdp);

  streamer.close();
  viewer.close();
});

test("relays sdp_answer from viewer to streamer", async () => {
  const streamer = await registerAs("streamer");
  const viewer   = await registerAs("viewer");

  const { viewer_id } = await nextMessage(streamer);

  const answerSdp = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\n";
  viewer.send(JSON.stringify({ type: "sdp_answer", viewer_id, sdp: answerSdp }));

  const received = await nextMessage(streamer);
  assert.equal(received.type, "sdp_answer");
  assert.equal(received.sdp, answerSdp);

  streamer.close();
  viewer.close();
});

test("relays ice_candidate from streamer to viewer", async () => {
  const streamer = await registerAs("streamer");
  const viewer   = await registerAs("viewer");
  const { viewer_id } = await nextMessage(streamer);

  const candidate = { candidate: "candidate:1 1 udp ...", sdpMid: "0", sdpMLineIndex: 0 };
  streamer.send(JSON.stringify({ type: "ice_candidate", viewer_id, candidate }));

  const received = await nextMessage(viewer);
  assert.equal(received.type, "ice_candidate");
  assert.deepEqual(received.candidate, candidate);

  streamer.close();
  viewer.close();
});

test("relays ice_candidate from viewer to streamer", async () => {
  const streamer = await registerAs("streamer");
  const viewer   = await registerAs("viewer");
  const { viewer_id } = await nextMessage(streamer);

  const candidate = { candidate: "candidate:2 1 udp ...", sdpMid: "0", sdpMLineIndex: 0 };
  viewer.send(JSON.stringify({ type: "ice_candidate", viewer_id, candidate }));

  const received = await nextMessage(streamer);
  assert.equal(received.type, "ice_candidate");
  assert.deepEqual(received.candidate, candidate);

  streamer.close();
  viewer.close();
});

test("notifies viewer when streamer disconnects", async () => {
  const streamer = await registerAs("streamer");
  const viewer   = await registerAs("viewer");

  // Drain the viewer_connected notification from the streamer
  await nextMessage(streamer);

  streamer.close();

  const msg = await nextMessage(viewer);
  assert.equal(msg.type, "streamer_disconnected");

  viewer.close();
});
