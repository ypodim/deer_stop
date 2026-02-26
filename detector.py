"""Hailo inference engine, shared frame buffer, and clip recorder."""

import csv
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Must be set before hailo_platform is imported so the runtime registers with
# the hailortcli monitor daemon.
os.environ.setdefault("HAILO_MONITOR", "1")

import cv2
import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
)
from sort import Sort

import monitor
import reviews as reviews_mod

_TZ_PST = ZoneInfo("America/Los_Angeles")


def _transcode(src: Path):
    """Re-encode a clip to H.264 in a background thread so the detector isn't stalled."""
    tmp = src.with_suffix(".tmp.mp4")
    t0 = time.monotonic()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264",
         "-preset", "fast", "-crf", "23", str(tmp)],
        capture_output=True,
    )
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        tmp.replace(src)
        print(f"Transcode done in {elapsed:.1f}s: {src.name}")
    else:
        tmp.unlink(missing_ok=True)
        print(f"ffmpeg transcode failed for {src} ({elapsed:.1f}s): {result.stderr.decode()}")


# ---------------------------------------------------------------------------
# COCO class table
# ---------------------------------------------------------------------------

COCO_CLASSES = {
    "person": 100,
    "bicycle": 1,
    "car": 0,
    "motorcycle": 1,
    "airplane": 1,
    "bus": 1,
    "train": 0,
    "truck": 0,
    "boat": 1,
    "traffic light": 0,
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

# Module-level CSV writer; initialised inside run()
_csv_writer = None
_csv_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Video source helpers
# ---------------------------------------------------------------------------

def _source_type(source: str | None) -> str:
    if source is None:
        return "usb"
    if source.lower().startswith("rtsp://"):
        return "rtsp"
    return "file"


def open_source(source: str | None) -> cv2.VideoCapture:
    kind = _source_type(source)

    if kind == "usb":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam /dev/video0")
            sys.exit(1)
        return cap

    if kind == "rtsp":
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


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Thread-safe single-frame store; producer calls update(), consumers call get_or_none()."""

    def __init__(self):
        self.frame: bytes | None = None
        self.lock = threading.Lock()
        self.event = threading.Event()

    def update(self, frame: np.ndarray):
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            self.frame = jpeg.tobytes()
        self.event.set()

    def get(self) -> bytes | None:
        """Block until first frame is available, then return latest."""
        self.event.wait()
        with self.lock:
            return self.frame

    def get_or_none(self) -> bytes | None:
        """Non-blocking; returns None if no frame has arrived yet."""
        with self.lock:
            return self.frame


# ---------------------------------------------------------------------------
# ClipRecorder
# ---------------------------------------------------------------------------

class ClipRecorder:
    """Records short video clips bracketing high-priority detections."""

    IDLE = "idle"
    RECORDING = "recording"

    def __init__(self, clips_dir: Path, fps: float, width: int, height: int,
                 pre_roll: float = 3.0, post_roll: float = 5.0, max_duration: float = 120.0):
        self.clips_dir = clips_dir
        self.fps = max(fps, 1.0)
        self.width = width
        self.height = height
        self.post_roll = post_roll
        self.max_duration = max_duration

        self.state = self.IDLE
        self._pre_buf: deque = deque(maxlen=max(1, int(self.fps * pre_roll)))
        self._writer: cv2.VideoWriter | None = None
        self._clip_id: str | None = None
        self._clip_path: Path | None = None
        self._thumb_path: Path | None = None
        self._start_time: float = 0.0
        self._last_det_time: float = 0.0
        self._best_conf: float = 0.0
        self._best_class: str | None = None
        self._thumb_saved: bool = False

    def push(self, frame: np.ndarray, det_class: str | None = None,
             det_conf: float = 0.0) -> dict | None:
        """
        Feed a frame; supply det_class/det_conf when a high-priority detection fired.
        Returns a clip-info dict when a clip closes, otherwise None.
        """
        now = time.time()

        if self.state == self.IDLE:
            self._pre_buf.append(frame.copy())
            if det_class is not None:
                self._start(det_class, det_conf, now)
        else:  # RECORDING
            self._writer.write(frame)
            if not self._thumb_saved:
                cv2.imwrite(str(self._thumb_path), frame)
                self._thumb_saved = True
            if now - self._start_time >= self.max_duration:
                return self._close()
            if det_class is not None:
                self._last_det_time = now
                if det_conf > self._best_conf:
                    self._best_conf = det_conf
                    self._best_class = det_class
            elif now - self._last_det_time > self.post_roll:
                return self._close()

        return None

    def force_close(self) -> dict | None:
        """Close any open clip immediately (e.g. on shutdown)."""
        if self.state == self.RECORDING:
            return self._close()
        return None

    def _start(self, class_name: str, conf: float, now: float):
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self._clip_id = f"{ts}_{class_name}"
        self._clip_path = self.clips_dir / f"{self._clip_id}.mp4"
        self._thumb_path = self.clips_dir / f"{self._clip_id}.jpg"
        self._start_time = now
        self._last_det_time = now
        self._best_conf = conf
        self._best_class = class_name
        self._thumb_saved = False
        self.state = self.RECORDING

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self._clip_path), fourcc, self.fps, (self.width, self.height)
        )
        for f in self._pre_buf:
            self._writer.write(f)
        self._pre_buf.clear()

    def _close(self) -> dict:
        self._writer.release()
        self._writer = None
        self.state = self.IDLE
        threading.Thread(target=_transcode, args=(self._clip_path,), daemon=True).start()
        return {
            "id": self._clip_id,
            "path": str(self._clip_path),
            "thumb": str(self._thumb_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "class_name": self._best_class,
            "confidence": round(float(self._best_conf), 4),
            "reviewed": False,
        }


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing / drawing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, input_height: int, input_width: int) -> tuple:
    orig_h, orig_w = frame.shape[:2]
    scale = min(input_width / orig_w, input_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (input_width - new_w) // 2
    pad_h = (input_height - new_h) // 2
    padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded, (scale, pad_w, pad_h, orig_w, orig_h)


def _tile_positions(length: int, tile: int, step: int) -> list[int]:
    """1-D start positions for tiles covering [0, length)."""
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile, step))
    if not positions or positions[-1] + tile < length:
        positions.append(length - tile)
    return positions


def make_tiles(frame: np.ndarray, tile_size: int, overlap: float) -> list[tuple]:
    """Split frame into overlapping tile_size×tile_size crops.

    Returns list of (tile_img, x_offset, y_offset).
    Edge tiles are zero-padded when the frame is smaller than tile_size.
    """
    h, w = frame.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))
    tiles = []
    for y in _tile_positions(h, tile_size, step):
        for x in _tile_positions(w, tile_size, step):
            crop = frame[y:y + tile_size, x:x + tile_size]
            ch, cw = crop.shape[:2]
            if ch < tile_size or cw < tile_size:
                padded = np.full((tile_size, tile_size, 3), 114, dtype=np.uint8)
                padded[:ch, :cw] = crop
                crop = padded
            else:
                crop = np.ascontiguousarray(crop)
            tiles.append((crop, x, y))
    return tiles


def _compute_downscale(frame_w: int, frame_h: int,
                       tile_size: int, overlap: float,
                       max_tiles: int) -> tuple[int, int]:
    """Return (w, h) scaled down so that make_tiles produces at most max_tiles.

    If the frame already fits within the limit, the original dimensions are returned.
    Uses binary search over the scale factor to find the largest resolution that
    still satisfies the constraint.
    """
    step = max(1, int(tile_size * (1 - overlap)))

    def n_tiles(w, h):
        return (len(_tile_positions(w, tile_size, step)) *
                len(_tile_positions(h, tile_size, step)))

    if n_tiles(frame_w, frame_h) <= max_tiles:
        return frame_w, frame_h

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if n_tiles(max(1, int(frame_w * mid)), max(1, int(frame_h * mid))) <= max_tiles:
            lo = mid
        else:
            hi = mid

    return max(1, int(frame_w * lo)), max(1, int(frame_h * lo))


def _log_detection(class_name, class_id, conf, x1, y1, x2, y2):
    if _csv_writer is None:
        return
    ts = datetime.now(_TZ_PST).isoformat()
    with _csv_lock:
        _csv_writer.writerow([ts, class_name, class_id, f"{conf:.4f}",
                              f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])


def postprocess(outputs: dict, scale_info: tuple, conf_thresh: float) -> list:
    """
    Post-process Hailo NMS output. Returns [(x1, y1, x2, y2, conf, class_id), ...].
    """
    scale, pad_w, pad_h, orig_w, orig_h = scale_info
    input_h = int(orig_h * scale + 2 * pad_h)
    input_w = int(orig_w * scale + 2 * pad_w)

    output = outputs[list(outputs.keys())[0]]

    if not (isinstance(output, list) and len(output) > 0):
        print("ERROR: Unexpected output format")
        return []

    class_outputs = output[0] if isinstance(output[0], list) else output
    boxes = []

    for class_id, class_detections in enumerate(class_outputs):
        class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"
        if COCO_CLASSES.get(class_name, 0) == 0:
            continue
        if not (isinstance(class_detections, np.ndarray) and class_detections.size > 0):
            continue

        for det in class_detections:
            y1_norm, x1_norm, y2_norm, x2_norm, conf = det
            if conf < conf_thresh:
                continue

            x1 = max(0, min((x1_norm * input_w - pad_w) / scale, orig_w))
            y1 = max(0, min((y1_norm * input_h - pad_h) / scale, orig_h))
            x2 = max(0, min((x2_norm * input_w - pad_w) / scale, orig_w))
            y2 = max(0, min((y2_norm * input_h - pad_h) / scale, orig_h))

            # print(f"  {class_name} ({class_id}): {conf:.2f}  {x2-x1:.0f}x{y2-y1:.0f}")

            boxes.append([x1, y1, x2, y2, conf, class_id])

    return boxes


def nms(detections: list, iou_threshold: float = 0.5) -> list:
    """Greedy NMS — removes duplicates from overlapping tiles."""
    if len(detections) <= 1:
        return detections
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if _iou(best[0], best[1], best[2], best[3],
                    d[0],    d[1],    d[2],    d[3]) < iou_threshold
        ]
    return kept


def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det[:6]
        track_id = int(det[6]) if len(det) > 6 else None
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
        color = ((class_id * 41) % 255, (class_id * 72) % 255, (class_id * 113) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = COCO_NAMES[class_id]
        if track_id is not None:
            label += f" #{track_id}"
        label += f": {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# ---------------------------------------------------------------------------
# Main inference loop (runs in a daemon thread)
# ---------------------------------------------------------------------------

def run(frame_buffer: FrameBuffer, stop_event: threading.Event, args,
        reviews_path: Path, reviews_lock: threading.Lock, stats_monitor=None):
    """Blocking inference loop. Intended to run in a daemon thread."""
    global _csv_writer

    source = args.source
    kind = _source_type(source)

    # Set up CSV detection log
    log_path = Path(args.log)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    csv_file = open(log_path, "a", newline="")
    _csv_writer = csv.writer(csv_file)
    if write_header:
        _csv_writer.writerow(["timestamp", "class_name", "class_id", "confidence",
                              "x1", "y1", "x2", "y2"])
        csv_file.flush()

    print(f"Loading HEF: {args.model}")
    hef = HEF(str(args.model))
    input_vstream_infos = hef.get_input_vstream_infos()
    output_vstream_infos = hef.get_output_vstream_infos()

    input_shape = input_vstream_infos[0].shape
    input_height, input_width = input_shape[0], input_shape[1]
    input_name = input_vstream_infos[0].name
    output_name = output_vstream_infos[0].name
    print(f"Model input: {input_width}x{input_height}")

    with VDevice() as device:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        # Internal Hailo hardware monitor — publishes device utilization and FPS
        # into the shared stats store so the web UI can display them.
        _hailo_mon = monitor.StatsMonitor()

        cap = open_source(source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_time = 1.0 / fps

        label = source_label(source)
        print(f"Source: {label} ({width}x{height} @ {fps:.2f} FPS)")

        # Scale frame down if needed so tile count stays within the batch budget.
        max_tiles = args.batch_size - 1  # reserve one slot for the full-frame pass
        if args.tile_overlap > 0:
            proc_w, proc_h = _compute_downscale(
                width, height, input_height, args.tile_overlap, max_tiles,
            )
            if proc_w != width or proc_h != height:
                print(f"Downscale: {width}x{height} → {proc_w}x{proc_h} "
                      f"to keep tiles ≤ {max_tiles}")
        else:
            proc_w, proc_h = width, height

        if args.tile_overlap > 0:
            sample_tiles = make_tiles(
                np.zeros((proc_h, proc_w, 3), dtype=np.uint8),
                input_height, args.tile_overlap,
            )
            n_total = len(sample_tiles) + 1  # tiles + full-frame
            print(f"Tiling: {len(sample_tiles)} tiles + 1 full-frame = {n_total} inferences/frame "
                  f"({input_height}×{input_width}, {args.tile_overlap:.0%} overlap, batch={args.batch_size})")
        else:
            print(f"Single-pass: 1 inference/frame (batch={args.batch_size})")

        clips_dir = Path(args.clips_dir)
        recorder = ClipRecorder(
            clips_dir, fps, proc_w, proc_h,
            pre_roll=args.pre_roll, post_roll=args.post_roll, max_duration=args.max_clip,
        )
        tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

        _EMA_ALPHA = 0.1
        _ema_tile_fps: float | None = None
        _ema_frame_ms: float | None = None

        try:
            with network_group.activate():
                with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:

                    while not stop_event.is_set():
                        start = time.time()

                        ret, frame = cap.read()
                        if not ret:
                            if kind in ("usb", "rtsp"):
                                if kind == "rtsp":
                                    print("RTSP: lost connection, reconnecting...")
                                    cap.release()
                                    time.sleep(2)
                                    cap = open_source(source)
                                continue
                            if args.loop:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            break

                        if proc_w != width or proc_h != height:
                            frame = cv2.resize(frame, (proc_w, proc_h))

                        # Build batch: tiles (if tiling) + full-frame letterbox + zero padding
                        if args.tile_overlap > 0:
                            tiles = make_tiles(frame, input_height, args.tile_overlap)
                            tile_scale = (1.0, 0, 0, input_height, input_width)
                            batch_imgs = [t[0] for t in tiles]
                        else:
                            tiles = []
                            tile_scale = None
                            batch_imgs = []

                        full_input, full_scale_info = preprocess(frame, input_height, input_width)
                        batch_imgs.append(full_input)

                        dummy = np.zeros((input_height, input_width, 3), dtype=np.uint8)
                        batch_imgs.extend([dummy] * (args.batch_size - len(batch_imgs)))

                        t_tile_start = time.monotonic()
                        outputs = pipeline.infer({input_name: np.stack(batch_imgs)})
                        out_list = outputs[output_name]
                        t_tile_elapsed = time.monotonic() - t_tile_start

                        # Collect tile detections (translate from tile-space to frame-space)
                        all_dets = []
                        for i, (_, tx, ty) in enumerate(tiles):
                            tile_out = {output_name: [out_list[i]]}
                            for d in postprocess(tile_out, tile_scale, args.conf):
                                d[0] += tx; d[2] += tx
                                d[1] += ty; d[3] += ty
                                all_dets.append(d)

                        # Full-frame output (index = number of tiles)
                        full_out = {output_name: [out_list[len(tiles)]]}
                        all_dets.extend(postprocess(full_out, full_scale_info, args.conf))

                        detections = nms(all_dets)

                        if t_tile_elapsed > 0 and stats_monitor is not None:
                            n_infer = len(tiles) + 1
                            raw_tile_fps = n_infer / t_tile_elapsed
                            raw_frame_ms = t_tile_elapsed * 1000
                            if _ema_tile_fps is None:
                                _ema_tile_fps = raw_tile_fps
                                _ema_frame_ms = raw_frame_ms
                            else:
                                _ema_tile_fps = _EMA_ALPHA * raw_tile_fps + (1 - _EMA_ALPHA) * _ema_tile_fps
                                _ema_frame_ms = _EMA_ALPHA * raw_frame_ms + (1 - _EMA_ALPHA) * _ema_frame_ms
                            stats_monitor.update({
                                **_hailo_mon.get(),
                                "tile_fps": round(_ema_tile_fps, 1),
                                "frame_ms": round(_ema_frame_ms, 1),
                            })

                        # Log priority-100 detections with frame-space coordinates
                        for d in detections:
                            if COCO_CLASSES.get(COCO_NAMES[int(d[5])], 0) == 100:
                                _log_detection(COCO_NAMES[int(d[5])], int(d[5]),
                                               d[4], d[0], d[1], d[2], d[3])
                        if detections:
                            csv_file.flush()

                        # Run SORT tracker; recover class_id by IoU-matching back to raw detections
                        sort_in = np.array([[*d[:5]] for d in detections], dtype=float) \
                                  if detections else np.empty((0, 5))
                        tracked = tracker.update(sort_in)
                        tracked_dets = []
                        for trk in tracked:
                            tx1, ty1, tx2, ty2, tid = trk
                            best_iou, best_det = 0.0, None
                            for det in detections:
                                iou = _iou(tx1, ty1, tx2, ty2, *det[:4])
                                if iou > best_iou:
                                    best_iou, best_det = iou, det
                            if best_det is not None:
                                x1, y1, x2, y2, conf, class_id = best_det[:6]
                                tracked_dets.append([x1, y1, x2, y2, conf, class_id, int(tid)])
                        detections = tracked_dets

                        annotated = draw_detections(frame.copy(), detections)
                        frame_buffer.update(annotated)

                        # Feed clip recorder; trigger on priority-100 classes
                        high = [d for d in detections
                                if COCO_CLASSES.get(COCO_NAMES[int(d[5])], 0) == 100]
                        if high:
                            best = max(high, key=lambda d: d[4])
                            clip_info = recorder.push(annotated, COCO_NAMES[int(best[5])], best[4])
                        else:
                            clip_info = recorder.push(annotated)

                        if clip_info is not None:
                            reviews_mod.add(reviews_path, reviews_lock, clip_info)

                        elapsed = time.time() - start
                        if kind != "rtsp" and elapsed < frame_time:
                            time.sleep(frame_time - elapsed)

        finally:
            clip_info = recorder.force_close()
            if clip_info is not None:
                reviews_mod.add(reviews_path, reviews_lock, clip_info)
            cap.release()
            csv_file.close()
