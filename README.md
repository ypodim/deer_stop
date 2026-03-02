# DeerStop

Wildlife camera pipeline: RTSP/USB video → tiled YOLO inference → MJPEG stream + clip recorder, with an iOS app for remote access.

Supports two inference backends:
- **Hailo-8** accelerator (Raspberry Pi / embedded)
- **Nvidia GPU** via ultralytics YOLO (`.pt` or `.engine`)

---

## Repository structure

```
detection/      Detection server — Python pipeline (runs on edge hardware)
node/           Node server — WebRTC signaling, TURN relay, nginx config
ios/            iOS app — SwiftUI client (live stream, clip review, notifications)
```

---

## Detection server

### Quick start

```bash
source /path/to/venv/bin/activate
# edit detection/settings.toml (and optionally detection/settings.local.toml for secrets)
cd detection && python stream_yolo.py
```

Then open:
- `http://localhost:8080` — live annotated MJPEG stream
- `http://localhost:8080/review` — clip review UI
- `http://localhost:8080/stats` — JSON stats (fps, GPU/CPU utilization)
- `http://localhost:8080/events` — SSE stream of clip-saved events

### settings.toml

```toml
[server]
port = 8080
host = "127.0.0.1"   # 0.0.0.0 for direct LAN, 127.0.0.1 when using autossh tunnel

[source]
url = "rtsp://..."   # omit for USB webcam (/dev/video0), or set to a file path
loop = false

[model]
backend = "nvidia"   # "hailo" or "nvidia"
model = "/path/to/yolo11l.pt"
conf = 0.6
tile_overlap = 0.5
batch_size = 8       # Hailo only
# imgsz = 640        # Nvidia only

[node]
signaling_url = "wss://node.polychronis.gr/signaling"
turn_url = "turn:node.polychronis.gr:3478"
auth_token = ""      # put real value in settings.local.toml (gitignored)

[recording]
clips_dir = "/path/to/clips"
pre_roll = 3.0
post_roll = 5.0
max_clip = 30.0
```

### Systemd services

```bash
# Detection pipeline
sudo cp detection/deerstop.service /etc/systemd/system/
sudo systemctl enable --now deerstop

# autossh reverse tunnel (forwards Node:18080 → localhost:8080)
sudo cp detection/deerstop-tunnel.service /etc/systemd/system/
sudo systemctl enable --now deerstop-tunnel
```

### Tests

```bash
cd detection
python -m pytest tests/
```

---

## Node server

Runs WebRTC signaling and TURN relay on `node.polychronis.gr`.

### Setup

```bash
cd node
cp .env.example .env        # fill in TURN_SECRET, AUTH_TOKEN, TURN_URL
cp coturn/turnserver.conf.example coturn/turnserver.conf   # fill in TURN_SECRET
docker-compose up -d

# Install nginx config (uses existing nginx + certbot on Node)
sudo cp nginx/deerstop.conf /etc/nginx/conf.d/deerstop.conf
# Create basic-auth password file for the REST API
sudo htpasswd -c /etc/nginx/.deerstop.htpasswd <username>
sudo nginx -t && sudo systemctl reload nginx
```

### Firewall

Open on Node: `443/tcp`, `3478/tcp+udp`, `49152-65535/udp`.

### Tests

```bash
cd node/signaling
npm install
npm test
```

---

## iOS app

SwiftUI app targeting iOS 17+. Streams live video via WebRTC, browses and reviews clips, and receives local notifications on new detections.

### Build

1. Open `ios/Package.swift` in Xcode (File → Open) **or** add the WebRTC SPM package manually via *File → Add Package Dependencies* using `https://github.com/stasel/WebRTC.git`.
2. Build and run on a device or simulator.
3. In the **Settings** tab, enter your Node server URL and auth token.

### Key dependencies

- [`stasel/WebRTC`](https://github.com/stasel/WebRTC) — Google's precompiled WebRTC iOS framework (via Swift Package Manager)

---

## Tiling

Frames are split into overlapping `input_size × input_size` tiles, each inferred independently, then results are merged with NMS. The full frame is also inferred in a single pass.

- **Hailo**: tile count is capped by `batch_size - 1`. Frame is downscaled if needed.
- **Nvidia**: no tile cap.

Set `tile_overlap = 0.0` to disable tiling.

## Detection classes

COCO classes are assigned a priority in `detection/detector.py`:

| Priority | Behaviour | Examples |
|----------|-----------|---------|
| `0` | Ignored | car, bench, chair |
| `1` | Annotated, not recorded | bird, dog, bicycle |
| `100` | Annotated + clip recorded + SSE event | person, sheep, cow, giraffe |

## Clip recording

When a priority-100 detection occurs, the recorder writes a clip (pre-roll + post-roll) to `clips_dir`. Clips are re-encoded to H.264 in a background thread via ffmpeg. A clip-saved event is broadcast to all SSE subscribers.
