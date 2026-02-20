#!/usr/bin/env python3
"""
YOLO detection on Hailo with MJPEG streaming and clip review.

Usage:
    python stream_yolo.py                                    # USB webcam (default)
    python stream_yolo.py --source rtsp://user:pass@ip/path  # RTSP stream
    python stream_yolo.py --source video.mp4 --loop          # video file

    http://localhost:8080        live stream
    http://localhost:8080/review detection review
"""

import argparse
import sys
import threading
from pathlib import Path

import tornado.ioloop

import detector
import reviews
import web

HEF_PATH = "/home/pol/ws/deerstop/yolov11m.hef"
CLIPS_DIR = "/home/pol/ws/deerstop/clips"


def main():
    parser = argparse.ArgumentParser(description="YOLO detection on Hailo with MJPEG streaming")
    parser.add_argument("--source", type=str, default=None,
                        help="USB webcam (omit), rtsp://... stream, or path to video file")
    parser.add_argument("--video", type=str, default=None,
                        help="(deprecated, use --source) path to video file")
    parser.add_argument("--hef", type=str, default=HEF_PATH, help="Path to HEF model file")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--loop", action="store_true", help="Loop video file")
    parser.add_argument("--log", type=str, default="detections.log", help="CSV log file path")
    parser.add_argument("--clips-dir", type=str, default=CLIPS_DIR,
                        help="Directory for clip recordings")
    parser.add_argument("--reviews", type=str, default="reviews.json",
                        help="Path to reviews JSON file")
    parser.add_argument("--pre-roll", type=float, default=3.0,
                        help="Seconds of footage to keep before a detection")
    parser.add_argument("--post-roll", type=float, default=5.0,
                        help="Seconds to keep recording after last detection")
    args = parser.parse_args()

    # Backwards compat: --video → --source
    if args.source is None and args.video is not None:
        args.source = args.video

    hef_path = Path(args.hef)
    if not hef_path.exists():
        print(f"Error: HEF file not found: {hef_path}")
        sys.exit(1)

    reviews_path = Path(args.reviews)
    reviews_lock = threading.Lock()
    clips_dir = Path(args.clips_dir).resolve()
    templates_dir = Path(__file__).parent / "templates"

    clips_dir.mkdir(parents=True, exist_ok=True)

    # Drop reviews whose files are gone
    reviews.prune(reviews_path, reviews_lock)

    # Shared frame buffer
    frame_buffer = detector.FrameBuffer()

    # Start inference in background
    stop_event = threading.Event()
    det_thread = threading.Thread(
        target=detector.run,
        args=(frame_buffer, stop_event, args, reviews_path, reviews_lock),
        daemon=True,
    )
    det_thread.start()

    # Build and start Tornado
    app = web.make_app(frame_buffer, reviews_path, reviews_lock, clips_dir, templates_dir)
    app.listen(args.port)

    print(f"Stream:  http://localhost:{args.port}")
    print(f"Review:  http://localhost:{args.port}/review")
    print("Press Ctrl+C to stop")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        det_thread.join(timeout=5)

    print("Done")


if __name__ == "__main__":
    main()
