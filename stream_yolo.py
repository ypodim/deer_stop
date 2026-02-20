#!/usr/bin/env python3
"""
Video file YOLO detection on Hailo with HTTP MJPEG streaming output.

Requirements:
    pip install opencv-python numpy
    Hailo RT installed (hailort Python bindings)

Usage:
    python stream_yolo.py                                    # USB webcam (default)
    python stream_yolo.py --source rtsp://user:pass@ip/path  # RTSP stream
    python stream_yolo.py --source video.mp4 --loop          # video file
    Then open http://localhost:8080 in a browser or VLC
"""

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType


# COCO class names for YOLOv8
COCO_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 0,
    "motorcycle": 1,
    "airplane": 1,
    "bus": 1,
    "train": 0,
    "truck": 0,
    "boat": 1,
    "traffic light": 1,
    "fire hydrant": 1,
    "stop sign": 1,
    "parking meter": 1,
    "bench": 0,
    "bird": 1,
    "cat": 1,
    "dog": 1,
    "horse": 1,
    "sheep": 100,
    "cow": 100,
    "elephant": 1,
    "bear": 1,
    "zebra": 1,
    "giraffe": 100,
    "backpack": 1,
    "umbrella": 1,
    "handbag": 1,
    "tie": 1,
    "suitcase": 1,
    "frisbee": 1,
    "skis": 1,
    "snowboard": 1,
    "sports ball": 1,
    "kite": 1,
    "baseball bat": 1,
    "baseball glove": 1,
    "skateboard": 1,
    "surfboard": 1,
    "tennis racket": 1,
    "bottle": 1,
    "wine glass": 1,
    "cup": 1,
    "fork": 1,
    "knife": 1,
    "spoon": 1,
    "bowl": 1,
    "banana": 1,
    "apple": 1,
    "sandwich": 1,
    "orange": 1,
    "broccoli": 1,
    "carrot": 1,
    "hot dog": 1,
    "pizza": 1,
    "donut": 1,
    "cake": 1,
    "chair": 0,
    "couch": 1,
    "potted plant": 0,
    "bed": 1,
    "dining table": 1,
    "toilet": 1,
    "tv": 1,
    "laptop": 1,
    "mouse": 1,
    "remote": 1,
    "keyboard": 1,
    "cell phone": 1,
    "microwave": 1,
    "oven": 1,
    "toaster": 1,
    "sink": 1,
    "refrigerator": 1,
    "book": 1,
    "clock": 1,
    "vase": 1,
    "scissors": 1,
    "teddy bear": 1,
    "hair drier": 1,
    "toothbrush": 1,
}
COCO_NAMES = list(COCO_CLASSES.keys())

# Global CSV writer for detection logging (set up in main)
_csv_writer = None
_csv_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Video source helpers
# ---------------------------------------------------------------------------

def _source_type(source: str | None) -> str:
    """Return 'usb', 'rtsp', or 'file' based on the source string."""
    if source is None:
        return "usb"
    if source.lower().startswith("rtsp://"):
        return "rtsp"
    return "file"


def open_source(source: str | None) -> cv2.VideoCapture:
    """Open a VideoCapture for USB camera, RTSP stream, or video file."""
    kind = _source_type(source)

    if kind == "usb":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam /dev/video0")
            sys.exit(1)
        return cap

    if kind == "rtsp":
        # Use TCP transport for reliability; small buffer to reduce latency
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream: {source}")
            sys.exit(1)
        return cap

    # file
    path = Path(source)
    if not path.exists():
        print(f"Error: Video file not found: {path}")
        sys.exit(1)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {path}")
        sys.exit(1)
    return cap


def source_label(source: str | None) -> str:
    kind = _source_type(source)
    if kind == "usb":
        return "webcam"
    if kind == "rtsp":
        return f"rtsp ({source})"
    return str(source)


class FrameBuffer:
    """Thread-safe frame buffer for sharing frames between producer and consumers."""

    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.event = threading.Event()

    def update(self, frame):
        with self.lock:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.frame = jpeg.tobytes()
        self.event.set()

    def get(self):
        self.event.wait()
        with self.lock:
            return self.frame


frame_buffer = FrameBuffer()
running = True


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="margin:0;background:#000;">
                <img src="/stream" style="width:100%;height:100%;object-fit:contain;">
                </body></html>
            """)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while running:
                    frame = frame_buffer.get()
                    if frame is None:
                        break
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def run_server(port: int):
    """Run the HTTP server."""
    server = HTTPServer(("0.0.0.0", port), MJPEGHandler)
    server.timeout = 1
    while running:
        server.handle_request()


def preprocess(frame: np.ndarray, input_height: int, input_width: int) -> tuple[np.ndarray, tuple]:
    """Preprocess frame for YOLO inference. Returns preprocessed image and scale info."""
    orig_h, orig_w = frame.shape[:2]

    # Calculate scale to fit in input size while maintaining aspect ratio
    scale = min(input_width / orig_w, input_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to input size (letterbox)
    pad_w = (input_width - new_w) // 2
    pad_h = (input_height - new_h) // 2
    padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, (scale, pad_w, pad_h, orig_w, orig_h)


def _log_detection(class_name, class_id, conf, x1, y1, x2, y2):
    """Write a detection row to the CSV log."""
    if _csv_writer is None:
        return
    ts = datetime.now(timezone.utc).isoformat()
    with _csv_lock:
        _csv_writer.writerow([ts, class_name, class_id, f"{conf:.4f}",
                              f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])


def postprocess(outputs: dict, scale_info: tuple, conf_thresh: float, iou_thresh: float = 0.45) -> list:
    """
    Post-process YOLO output. Returns list of detections: [(x1, y1, x2, y2, conf, class_id), ...]
    Handles Hailo's built-in NMS postprocess output format.
    """
    scale, pad_w, pad_h, orig_w, orig_h = scale_info
    input_h, input_w = int(orig_h * scale + 2 * pad_h), int(orig_w * scale + 2 * pad_w)

    output_name = list(outputs.keys())[0]
    output = outputs[output_name]

    # Handle Hailo NMS postprocess output: list of [list of arrays per class]
    # Each class array has shape (N, 5) with [y1, x1, y2, x2, conf] normalized
    if isinstance(output, list) and len(output) > 0:
        class_outputs = output[0] if isinstance(output[0], list) else output

        boxes = []
        for class_id, class_detections in enumerate(class_outputs):
            class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"
            if COCO_CLASSES[class_name] == 0:
                continue

            if isinstance(class_detections, np.ndarray) and class_detections.size > 0:
                for det in class_detections:
                    y1_norm, x1_norm, y2_norm, x2_norm, conf = det

                    if conf < conf_thresh:
                        continue

                    # Convert from normalized to input image coords
                    x1_inp = x1_norm * input_w
                    y1_inp = y1_norm * input_h
                    x2_inp = x2_norm * input_w
                    y2_inp = y2_norm * input_h

                    # Remove padding and scale back to original image
                    x1 = (x1_inp - pad_w) / scale
                    y1 = (y1_inp - pad_h) / scale
                    x2 = (x2_inp - pad_w) / scale
                    y2 = (y2_inp - pad_h) / scale

                    # Clip to image bounds
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))


                    print(f"  {class_name} ({class_id}): {conf:.2f}  {x2-x1:.0f}x{y2-y1:.0f}")

                    if COCO_CLASSES[class_name] == 100:
                        _log_detection(class_name, class_id, conf, x1, y1, x2, y2)

                    boxes.append([x1, y1, x2, y2, conf, class_id])

        return boxes

    print("ERROR: Unexpected output format")
    return []


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        # Color based on class
        color = ((class_id * 41) % 255, (class_id * 72) % 255, (class_id * 113) % 255)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{COCO_NAMES[class_id]}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def main():
    global running, _csv_writer
    HEF_PATH = "/home/pol/ws/deerstop/yolov11m.hef"

    parser = argparse.ArgumentParser(description="YOLO detection on Hailo with MJPEG streaming")
    parser.add_argument("--source", type=str, default=None,
                        help="Input source: omit for USB webcam, rtsp://... for RTSP stream, or path to video file")
    parser.add_argument("--video", type=str, default=None,
                        help="(deprecated, use --source) Path to input video file")
    parser.add_argument("--hef", type=str, default=HEF_PATH, help="Path to HEF model file")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--loop", action="store_true", help="Loop the video (ignored for webcam/RTSP)")
    parser.add_argument("--log", type=str, default="detections.log", help="CSV log file path")
    args = parser.parse_args()

    # --video is kept for backwards compat; --source takes priority
    source = args.source if args.source is not None else args.video
    kind = _source_type(source)

    hef_path = Path(args.hef)
    if not hef_path.exists():
        print(f"Error: HEF file not found: {hef_path}")
        sys.exit(1)

    # Set up CSV detection log
    log_path = Path(args.log)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    csv_file = open(log_path, "a", newline="")
    _csv_writer = csv.writer(csv_file)
    if write_header:
        _csv_writer.writerow(["timestamp", "class_name", "class_id", "confidence",
                              "x1", "y1", "x2", "y2"])
        csv_file.flush()

    # Load HEF and configure Hailo device
    print(f"Loading HEF: {args.hef}")
    hef = HEF(str(hef_path))

    # Get input shape from HEF
    input_vstream_infos = hef.get_input_vstream_infos()
    output_vstream_infos = hef.get_output_vstream_infos()

    input_shape = input_vstream_infos[0].shape
    input_height, input_width = input_shape[0], input_shape[1]
    print(f"Model input: {input_width}x{input_height}")

    # Create virtual device
    with VDevice() as device:
        # Configure network
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]

        # Create input/output params
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        # Open video source
        cap = open_source(source)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_time = 1.0 / fps

        label = source_label(source)
        print(f"Source: {label} ({width}x{height} @ {fps:.2f} FPS)")
        print(f"Detection log: {log_path}")
        print(f"Stream: http://localhost:{args.port}")
        print("Press Ctrl+C to stop")

        # Start HTTP server
        server_thread = threading.Thread(target=run_server, args=(args.port,), daemon=True)
        server_thread.start()

        # Run inference
        with network_group.activate():
            from hailo_platform import InferVStreams

            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                input_name = input_vstream_infos[0].name

                try:
                    while True:
                        start = time.time()

                        ret, frame = cap.read()
                        if not ret:
                            if kind in ("usb", "rtsp"):
                                if kind == "rtsp":
                                    # Reopen on RTSP disconnect
                                    print("RTSP: lost connection, reconnecting...")
                                    cap.release()
                                    time.sleep(2)
                                    cap = open_source(source)
                                continue
                            if args.loop:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            break

                        # Preprocess
                        input_data, scale_info = preprocess(frame, input_height, input_width)

                        # Run inference
                        input_dict = {input_name: np.expand_dims(input_data, axis=0)}
                        outputs = infer_pipeline.infer(input_dict)

                        # Postprocess
                        detections = postprocess(outputs, scale_info, args.conf)

                        # Draw results
                        annotated_frame = draw_detections(frame.copy(), detections)

                        frame_buffer.update(annotated_frame)

                        # Flush CSV after each frame with detections
                        if detections:
                            csv_file.flush()

                        elapsed = time.time() - start
                        # For RTSP, don't sleep — process as fast as possible
                        # to avoid buffering lag
                        if kind != "rtsp" and elapsed < frame_time:
                            time.sleep(frame_time - elapsed)

                except KeyboardInterrupt:
                    print("\nStopping...")
                finally:
                    running = False
                    cap.release()
                    csv_file.close()

    print("Done")


if __name__ == "__main__":
    main()
