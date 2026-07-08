#!/usr/bin/env python3
"""
YOLO detection with MJPEG streaming and clip review.

Configure via settings.toml in the same directory.

    http://localhost:8080        live stream
    http://localhost:8080/review detection review
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # pip install tomli
    except ModuleNotFoundError:
        sys.exit("tomllib not found; upgrade to Python 3.11+ or: pip install tomli")

import tornado.ioloop

import detector
import reviews
import stats
import web

SETTINGS_PATH = Path(__file__).parent / "settings.toml"
LOCAL_SETTINGS_PATH = Path(__file__).parent / "settings.local.toml"

DEFAULTS = {
    "server":    {"port": 8080, "host": "127.0.0.1"},
    "source":    {"loop": False},
    "model":     {"backend": "hailo", "model": str(Path(__file__).parent / "yolov11m.hef"), "conf": 0.5, "tile_overlap": 0.0, "batch_size": 1, "imgsz": 640},
    "recording": {
        "log":          "detections.log",
        "clips_dir":    str(Path(__file__).parent / "clips"),
        "reviews":      "reviews.json",
        "pre_roll":     3.0,
        "post_roll":    5.0,
        "max_clip":     30.0,
    },
    "node": {
        "signaling_url": "",
        "turn_url":      "",
        "auth_token":    "",
    },
    "notify": {
        "email": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, section by section."""
    merged = {**base}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def load_settings() -> SimpleNamespace:
    cfg = {}
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "rb") as f:
            cfg = tomllib.load(f)
    else:
        print(f"No settings.toml found at {SETTINGS_PATH}, using defaults")
    if LOCAL_SETTINGS_PATH.exists():
        with open(LOCAL_SETTINGS_PATH, "rb") as f:
            cfg = _deep_merge(cfg, tomllib.load(f))
        print(f"Loaded local overrides from {LOCAL_SETTINGS_PATH.name}")

    def get(section, key):
        return cfg.get(section, {}).get(key, DEFAULTS[section][key])

    model = get("model", "model")
    if not Path(model).exists():
        sys.exit(f"Error: Model file not found: {model}")

    # Cameras: prefer an array of [[camera]] tables (name + url); fall back to a
    # single [source] url for backward compatibility.
    cam_tables = cfg.get("camera")
    if cam_tables:
        cameras = [
            {"name": c.get("name", f"cam{i}"), "url": c.get("url")}
            for i, c in enumerate(cam_tables)
            if c.get("url")
        ]
    else:
        cameras = [{"name": "cam", "url": cfg.get("source", {}).get("url", None)}]

    return SimpleNamespace(
        source    = cameras[0]["url"],   # primary feed (used by WebRTC)
        cameras   = cameras,
        loop      = get("source", "loop"),
        backend      = get("model", "backend"),
        model        = model,
        conf         = get("model", "conf"),
        tile_overlap = get("model", "tile_overlap"),
        batch_size   = get("model", "batch_size"),
        imgsz        = get("model", "imgsz"),
        port      = get("server", "port"),
        host      = get("server", "host"),
        log       = get("recording", "log"),
        clips_dir = get("recording", "clips_dir"),
        reviews   = get("recording", "reviews"),
        pre_roll  = get("recording", "pre_roll"),
        post_roll = get("recording", "post_roll"),
        max_clip  = get("recording", "max_clip"),
        node_signaling_url = get("node", "signaling_url"),
        node_turn_url      = get("node", "turn_url"),
        node_auth_token    = get("node", "auth_token"),
        notify_email       = get("notify", "email"),
    )


def make_backend(args):
    """Create a fresh inference backend instance (one per camera)."""
    if args.backend == "nvidia":
        from backend_nvidia import NvidiaBackend
        return NvidiaBackend(args.model, args.conf)
    from backend_hailo import HailoBackend
    return HailoBackend(args.model, args.batch_size, args.conf)


def main():
    args = load_settings()

    reviews_path = Path(args.reviews)
    reviews_lock = threading.Lock()
    clips_dir = Path(args.clips_dir).resolve()
    templates_dir = Path(__file__).parent / "templates"

    clips_dir.mkdir(parents=True, exist_ok=True)

    reviews.prune(reviews_path, reviews_lock)

    event_queue = detector.EventQueue()
    stats_store = stats.StatsStore()

    # One independent pipeline per camera: its own backend + frame buffer +
    # detector thread. Reviews, events, and the notify/fan cooldown are shared.
    stop_event = threading.Event()
    streams: dict[str, detector.FrameBuffer] = {}
    det_threads = []
    for i, cam in enumerate(args.cameras):
        frame_buffer = detector.FrameBuffer()
        streams[cam["name"]] = frame_buffer
        # Stats reflect a single GPU; let the first camera own the stats store.
        stats_for_cam = stats_store if i == 0 else None
        t = threading.Thread(
            target=detector.run,
            args=(make_backend(args), frame_buffer, stop_event, args, reviews_path,
                  reviews_lock, stats_for_cam, event_queue, args.notify_email),
            kwargs={"source": cam["url"], "camera_name": cam["name"]},
            daemon=True,
        )
        t.start()
        det_threads.append(t)
        print(f"Camera '{cam['name']}': {cam['url']}")

    primary_name = args.cameras[0]["name"]
    primary_buffer = streams[primary_name]

    app = web.make_app(streams, reviews_path, reviews_lock, clips_dir, templates_dir, stats_store, event_queue)
    app.listen(args.port, address=args.host)

    print(f"Stream:  http://{args.host}:{args.port}          ({primary_name})")
    for name in list(streams)[1:]:
        print(f"Stream:  http://{args.host}:{args.port}/stream/{name}")
    print(f"Review:  http://{args.host}:{args.port}/review")

    # Start WebRTC signaling client if configured (primary camera only)
    if args.node_signaling_url and args.node_auth_token:
        import webrtc as webrtc_mod
        ioloop = tornado.ioloop.IOLoop.current()
        ioloop.asyncio_loop.create_task(
            webrtc_mod.run_signaling_client(primary_buffer, args)
        )
        print(f"Signaling: {args.node_signaling_url}")
    else:
        print("Node signaling not configured (set [node] signaling_url + auth_token)")

    print("Press Ctrl+C to stop")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        for t in det_threads:
            t.join(timeout=5)

    print("Done")


if __name__ == "__main__":
    main()
