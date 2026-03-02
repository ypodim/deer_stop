"""WebRTC signaling client and video track for remote live-stream access.

The detection server maintains a persistent outbound WebSocket connection to the
Node signaling server. When a viewer connects, this module creates an
RTCPeerConnection, attaches a video track fed from the FrameBuffer, and
completes the SDP/ICE handshake via the signaling server.

Requires: aiortc  (pip install aiortc)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from fractions import Fraction
from types import SimpleNamespace

import av
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType

from detector import FrameBuffer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Video track — pulls annotated JPEG frames from the shared FrameBuffer
# ---------------------------------------------------------------------------

class FrameBufferTrack(MediaStreamTrack):
    """aiortc VideoStreamTrack that sources frames from the detector FrameBuffer."""

    kind = "video"

    def __init__(self, frame_buffer: FrameBuffer):
        super().__init__()
        self._frame_buffer = frame_buffer
        self._pts = 0
        self._time_base = Fraction(1, 90000)  # standard RTP clock rate

    async def recv(self):
        loop = asyncio.get_event_loop()
        jpeg_bytes = await loop.run_in_executor(None, self._frame_buffer.get)

        codec = av.CodecContext.create("mjpeg", "r")
        packet = av.Packet(jpeg_bytes)
        frames = codec.decode(packet)
        if not frames:
            # Fallback: tiny blank frame to keep the track alive
            frame = av.VideoFrame(width=2, height=2, format="yuv420p")
        else:
            frame = frames[0].reformat(format="yuv420p")

        # Advance PTS at ~30 fps in the RTP clock domain
        self._pts += int(90000 / 30)
        frame.pts = self._pts
        frame.time_base = self._time_base
        return frame


# ---------------------------------------------------------------------------
# Peer connection lifecycle for a single viewer session
# ---------------------------------------------------------------------------

async def _handle_viewer(
    ws: ClientWebSocketResponse,
    viewer_id: str,
    frame_buffer: FrameBuffer,
    ice_servers: list[dict],
) -> None:
    """Create a peer connection for one viewer, exchange SDP/ICE, then keep alive."""
    pc = RTCPeerConnection()
    track = FrameBufferTrack(frame_buffer)
    pc.addTrack(track)

    @pc.on("icecandidate")
    async def on_ice(candidate):
        if candidate:
            await ws.send_json({
                "type": "ice_candidate",
                "viewer_id": viewer_id,
                "candidate": {
                    "candidate":     candidate.candidate,
                    "sdpMid":        candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                },
            })

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await ws.send_json({
        "type":      "sdp_offer",
        "viewer_id": viewer_id,
        "sdp":       pc.localDescription.sdp,
    })
    log.info("Sent SDP offer to viewer %s", viewer_id)
    return pc


# ---------------------------------------------------------------------------
# Persistent signaling client
# ---------------------------------------------------------------------------

async def run_signaling_client(frame_buffer: FrameBuffer, args: SimpleNamespace) -> None:
    """
    Persistent WebSocket client that connects to the Node signaling server,
    registers as the streamer, and handles viewer sessions.

    Reconnects with exponential backoff on failure.
    """
    url        = args.node_signaling_url
    token      = args.node_auth_token
    turn_url   = args.node_turn_url
    backoff    = 2.0
    max_backoff = 60.0

    while True:
        try:
            log.info("Connecting to signaling server: %s", url)
            async with ClientSession() as session:
                async with session.ws_connect(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    heartbeat=30,
                ) as ws:
                    backoff = 2.0  # reset on successful connect
                    log.info("Signaling connected")

                    # Register as the streamer
                    await ws.send_json({"type": "register", "role": "streamer"})

                    # Receive TURN credentials sent by the server on connect
                    ice_servers: list[dict] = []
                    if turn_url:
                        ice_servers.append({"urls": turn_url})  # credentials added below

                    # Active peer connections keyed by viewer_id
                    peer_connections: dict[str, RTCPeerConnection] = {}

                    async for msg in ws:
                        if msg.type == WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get("type")

                            if msg_type == "turn_credentials":
                                # Replace placeholder with credentialed TURN entry
                                ice_servers = [{
                                    "urls":       turn_url,
                                    "username":   data["username"],
                                    "credential": data["credential"],
                                }]
                                log.info("Received TURN credentials (user=%s)", data["username"])

                            elif msg_type == "viewer_connected":
                                viewer_id = data["viewer_id"]
                                log.info("Viewer connected: %s", viewer_id)
                                pc = await _handle_viewer(ws, viewer_id, frame_buffer, ice_servers)
                                peer_connections[viewer_id] = pc

                            elif msg_type == "sdp_answer":
                                viewer_id = data["viewer_id"]
                                pc = peer_connections.get(viewer_id)
                                if pc:
                                    await pc.setRemoteDescription(
                                        RTCSessionDescription(sdp=data["sdp"], type="answer")
                                    )
                                    log.info("Set remote SDP answer for viewer %s", viewer_id)

                            elif msg_type == "ice_candidate":
                                viewer_id = data["viewer_id"]
                                pc = peer_connections.get(viewer_id)
                                if pc and data.get("candidate"):
                                    c = data["candidate"]
                                    await pc.addIceCandidate(RTCIceCandidate(
                                        candidate=c["candidate"],
                                        sdpMid=c.get("sdpMid"),
                                        sdpMLineIndex=c.get("sdpMLineIndex"),
                                    ))

                            elif msg_type == "viewer_disconnected":
                                viewer_id = data["viewer_id"]
                                pc = peer_connections.pop(viewer_id, None)
                                if pc:
                                    await pc.close()
                                log.info("Viewer disconnected: %s", viewer_id)

                        elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                            break

                    # Clean up any open peer connections
                    for pc in peer_connections.values():
                        await pc.close()

        except Exception as exc:
            log.warning("Signaling error: %s — reconnecting in %.0fs", exc, backoff)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)
