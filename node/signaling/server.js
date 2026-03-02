/**
 * DeerStop WebRTC Signaling Server
 *
 * Brokers WebRTC session setup between the detection server (streamer) and
 * iOS app clients (viewers). Carries only tiny SDP/ICE messages — no video data.
 *
 * Environment variables (set in node/.env):
 *   PORT         Listening port (default: 3001)
 *   AUTH_TOKEN   Bearer token that all clients must present
 *   TURN_SECRET  Shared secret for generating time-limited TURN credentials
 *   TURN_URL     TURN server URL sent to clients (e.g. turn:node.polychronis.gr:3478)
 */

"use strict";

const { WebSocketServer, WebSocket } = require("ws");
const crypto = require("crypto");
const http = require("http");

const PORT        = parseInt(process.env.PORT || "3001", 10);
const AUTH_TOKEN  = process.env.AUTH_TOKEN  || "";
const TURN_SECRET = process.env.TURN_SECRET || "";
const TURN_URL    = process.env.TURN_URL    || "";

// ---------------------------------------------------------------------------
// TURN credential generation (standard coturn HMAC approach)
// ---------------------------------------------------------------------------

function makeTurnCredentials() {
  const ttl      = 24 * 3600;                          // valid for 24 h
  const username = `${Math.floor(Date.now() / 1000) + ttl}`;
  const hmac     = crypto.createHmac("sha1", TURN_SECRET);
  hmac.update(username);
  const credential = hmac.digest("base64");
  return { username, credential };
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

let streamer = null;                      // single streamer WebSocket
const viewers = new Map();               // viewer_id → WebSocket

function send(ws, obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ---------------------------------------------------------------------------
// HTTP server + WebSocket upgrade
// ---------------------------------------------------------------------------

const server = http.createServer((_req, res) => {
  res.writeHead(200);
  res.end("DeerStop signaling OK\n");
});

const wss = new WebSocketServer({ server });

wss.on("connection", (ws, req) => {
  // Authenticate via Bearer token
  const auth = req.headers["authorization"] || "";
  if (!AUTH_TOKEN || auth !== `Bearer ${AUTH_TOKEN}`) {
    ws.close(4401, "Unauthorized");
    return;
  }

  let role = null;
  let viewerId = null;

  // Send TURN credentials immediately after auth passes
  if (TURN_SECRET && TURN_URL) {
    const creds = makeTurnCredentials();
    send(ws, { type: "turn_credentials", ...creds, turn_url: TURN_URL });
  }

  ws.on("message", (raw) => {
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      return;
    }

    const type = data.type;

    // ---- Registration -------------------------------------------------------
    if (type === "register") {
      role = data.role;

      if (role === "streamer") {
        streamer = ws;
        console.log("[signaling] Streamer registered");
        // Notify any viewers already waiting
        for (const [id] of viewers) {
          send(streamer, { type: "viewer_connected", viewer_id: id });
        }

      } else if (role === "viewer") {
        viewerId = crypto.randomUUID();
        viewers.set(viewerId, ws);
        console.log(`[signaling] Viewer registered: ${viewerId}`);
        // Prompt streamer to send an offer for this viewer
        send(streamer, { type: "viewer_connected", viewer_id: viewerId });
      }
      return;
    }

    // ---- Relay: streamer → specific viewer ----------------------------------
    if (role === "streamer" && (type === "sdp_offer" || type === "ice_candidate")) {
      const target = viewers.get(data.viewer_id);
      if (target) send(target, data);
      return;
    }

    // ---- Relay: viewer → streamer -------------------------------------------
    if (role === "viewer" && (type === "sdp_answer" || type === "ice_candidate")) {
      send(streamer, { ...data, viewer_id: viewerId });
      return;
    }
  });

  ws.on("close", () => {
    if (role === "streamer") {
      console.log("[signaling] Streamer disconnected");
      streamer = null;
      // Notify all viewers
      for (const [id, vws] of viewers) {
        send(vws, { type: "streamer_disconnected" });
      }
    } else if (role === "viewer" && viewerId) {
      console.log(`[signaling] Viewer disconnected: ${viewerId}`);
      viewers.delete(viewerId);
      send(streamer, { type: "viewer_disconnected", viewer_id: viewerId });
    }
  });

  ws.on("error", (err) => {
    console.error("[signaling] WebSocket error:", err.message);
  });
});

// Only auto-listen when run directly (not when imported by tests)
if (require.main === module) {
  server.listen(PORT, "127.0.0.1", () => {
    console.log(`[signaling] Listening on 127.0.0.1:${PORT}`);
  });
}

module.exports = { server, wss }; // exported for tests
