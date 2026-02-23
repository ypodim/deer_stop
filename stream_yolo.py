#!/usr/bin/env python3
"""
YOLO detection on Hailo with MJPEG streaming and clip review.

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

import tornado.autoreload
import tornado.ioloop

import detector
import reviews
import web

SETTINGS_PATH = Path(__file__).parent / "settings.toml"

DEFAULTS = {
    "server":    {"port": 8080},
    "source":    {"loop": False},
    "model":     {"hef": str(Path(__file__).parent / "yolov11m.hef"), "conf": 0.5},
    "recording": {
        "log":          "detections.log",
        "clips_dir":    str(Path(__file__).parent / "clips"),
        "reviews":      "reviews.json",
        "pre_roll":     3.0,
        "post_roll":    5.0,
        "max_clip":     120.0,
    },
}


def load_settings() -> SimpleNamespace:
    cfg = {}
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "rb") as f:
            cfg = tomllib.load(f)
    else:
        print(f"No settings.toml found at {SETTINGS_PATH}, using defaults")

    def get(section, key):
        return cfg.get(section, {}).get(key, DEFAULTS[section][key])

    hef = get("model", "hef")
    if not Path(hef).exists():
        sys.exit(f"Error: HEF file not found: {hef}")

    return SimpleNamespace(
        source    = cfg.get("source", {}).get("url", None),
        loop      = get("source", "loop"),
        hef       = hef,
        conf      = get("model", "conf"),
        port      = get("server", "port"),
        log       = get("recording", "log"),
        clips_dir = get("recording", "clips_dir"),
        reviews   = get("recording", "reviews"),
        pre_roll  = get("recording", "pre_roll"),
        post_roll = get("recording", "post_roll"),
        max_clip  = get("recording", "max_clip"),
    )


def main():
    args = load_settings()

    reviews_path = Path(args.reviews)
    reviews_lock = threading.Lock()
    clips_dir = Path(args.clips_dir).resolve()
    templates_dir = Path(__file__).parent / "templates"

    clips_dir.mkdir(parents=True, exist_ok=True)

    reviews.prune(reviews_path, reviews_lock)

    frame_buffer = detector.FrameBuffer()

    stop_event = threading.Event()
    det_thread = threading.Thread(
        target=detector.run,
        args=(frame_buffer, stop_event, args, reviews_path, reviews_lock),
        daemon=True,
    )
    det_thread.start()

    app = web.make_app(frame_buffer, reviews_path, reviews_lock, clips_dir, templates_dir)
    app.listen(args.port)

    print(f"Stream:  http://localhost:{args.port}")
    print(f"Review:  http://localhost:{args.port}/review")
    print("Press Ctrl+C to stop")

    try:
        tornado.autoreload.start()
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        det_thread.join(timeout=5)

    print("Done")


if __name__ == "__main__":
    main()
