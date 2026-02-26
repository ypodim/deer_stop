# deerstop

Wildlife camera pipeline: RTSP/USB video → tiled YOLO inference → MJPEG stream + clip recorder.

Supports two inference backends:
- **Hailo-8** accelerator (Raspberry Pi / embedded)
- **Nvidia GPU** via ultralytics YOLO (`.pt` or `.engine`)

## Quick start

```bash
source /media/pol/pond/ws/deerstop_compile/venv/bin/activate
# edit settings.toml as needed
python stream_yolo.py
```

Then open:
- `http://localhost:8080` — live annotated MJPEG stream
- `http://localhost:8080/review` — clip review UI
- `http://localhost:8080/stats` — JSON stats (fps, GPU/CPU utilization)

## settings.toml

```toml
[server]
port = 8080

[source]
url = "rtsp://..."   # omit for USB webcam (/dev/video0), or set to a file path
loop = false         # loop video files (ignored for RTSP/webcam)

[model]
backend = "nvidia"   # "hailo" or "nvidia"
model = "/path/to/yolo11l.pt"   # .hef for hailo, .pt/.engine for nvidia
conf = 0.6           # confidence threshold
tile_overlap = 0.5   # 0.0 = single-pass, 0.5 = 50% overlap tiling
batch_size = 8       # Hailo only: must match HEF compiled batch size
# imgsz = 640        # Nvidia only: model input size (default 640)

[recording]
clips_dir = "/path/to/clips"
pre_roll = 3.0       # seconds of pre-detection footage to keep
post_roll = 5.0      # seconds to keep recording after last detection
max_clip = 30.0      # maximum clip length before forcing a new clip
```

## Tiling

Frames are split into overlapping `input_size × input_size` tiles, each inferred independently, then results are merged with NMS. The full frame is also inferred in a single pass. This improves detection of small objects.

- **Hailo**: tile count is capped by `batch_size - 1` (one slot reserved for full-frame). The frame is downscaled if needed to stay within the batch budget.
- **Nvidia**: no tile cap; all tiles + full frame are passed to `model.predict()` in one call.

Set `tile_overlap = 0.0` to disable tiling and run single-pass only.

## Detection classes

COCO classes are assigned a priority in `detector.py`:

| Priority | Behaviour | Examples |
|----------|-----------|---------|
| `0` | Ignored | car, bench, chair |
| `1` | Annotated, not recorded | bird, dog, bicycle |
| `100` | Annotated + clip recorded | person, sheep, cow, giraffe |

## Clip recording

When a priority-100 detection occurs, the recorder writes a clip (pre-roll + post-roll) to `clips_dir`. Clips are re-encoded to H.264 in a background thread via ffmpeg. A thumbnail is saved alongside each clip.

## File layout

```
stream_yolo.py      entry point
detector.py         inference loop, tiling, NMS, SORT tracker, clip recorder
backend_hailo.py    HailoBackend context manager
backend_nvidia.py   NvidiaBackend context manager
monitor.py          StatsMonitor (Hailo) + GpuStatsPoller (Nvidia, via nvidia-ml-py)
stats.py            StatsStore — thread-safe dict shared between detector and web server
web.py              Tornado handlers (stream, review, clips, stats)
reviews.py          Clip review persistence (reviews.json)
sort.py             SORT multi-object tracker
settings.toml       Configuration
templates/          HTML templates for the web UI
clips/              Recorded clips (created at runtime)
detections.log      CSV log of priority-100 detections
```

## Venv

`/media/pol/pond/ws/deerstop_compile/venv/`

Key packages: `torch` (CUDA), `ultralytics`, `tornado`, `opencv-python`, `psutil`, `nvidia-ml-py`, `scipy`.
