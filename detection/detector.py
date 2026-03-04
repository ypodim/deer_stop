"""Shared frame buffer, clip recorder, pre/postprocessing, and inference loop."""

import csv
import os
import queue as _queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from sort import Sort

import reviews as reviews_mod

_TZ_PST = ZoneInfo("America/Los_Angeles")


# ---------------------------------------------------------------------------
# Audio capture — continuous ring buffer from RTSP audio stream
# ---------------------------------------------------------------------------

class AudioCapture:
    """Continuously captures audio from an RTSP stream into a ring buffer."""

    def __init__(self, rtsp_url: str, buffer_seconds: float = 120.0,
                 sample_rate: int = 48000, channels: int = 1):
        self.rtsp_url = rtsp_url
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = 2  # s16le
        self.bytes_per_second = sample_rate * channels * self.bytes_per_sample

        self._chunk_duration = 0.5  # seconds per chunk
        self._chunk_bytes = int(self.bytes_per_second * self._chunk_duration)
        max_chunks = int(buffer_seconds / self._chunk_duration)
        self._buffer: deque[tuple[float, bytes]] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # -- public API --

    def start(self) -> bool:
        """Probe for audio and start capture. Returns True if audio is available."""
        if not self._has_audio():
            print("AudioCapture: no audio track found, continuing without audio")
            return False

        self._running = True
        self._process = subprocess.Popen(
            ["ffmpeg", "-rtsp_transport", "tcp",
             "-i", self.rtsp_url,
             "-vn",
             "-acodec", "pcm_s16le",
             "-ar", str(self.sample_rate),
             "-ac", str(self.channels),
             "-f", "s16le",
             "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="AudioCapture",
        )
        self._thread.start()
        print(f"AudioCapture: started ({self.sample_rate}Hz, {self.channels}ch, "
              f"buffer={self._buffer.maxlen * self._chunk_duration:.0f}s)")
        return True

    def extract(self, start_time: float, end_time: float,
                out_dir: Path) -> Path | None:
        """Extract audio for [start_time, end_time] and save as a WAV file."""
        with self._lock:
            chunks = [(t, d) for t, d in self._buffer
                      if t >= start_time - self._chunk_duration and t <= end_time]
        if not chunks:
            return None

        raw = b"".join(d for _, d in chunks)
        wav_path = out_dir / f"_audio_{int(start_time)}.wav"
        self._write_wav(wav_path, raw)
        return wav_path

    def restart(self):
        """Restart audio capture (e.g. after RTSP reconnection)."""
        self.stop()
        time.sleep(1)
        self.start()

    def stop(self):
        self._running = False
        self._cleanup_process()

    # -- internals --

    def _has_audio(self) -> bool:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet",
                 "-rtsp_transport", "tcp",
                 "-select_streams", "a",
                 "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0",
                 self.rtsp_url],
                capture_output=True, timeout=10,
            )
            return b"audio" in result.stdout
        except Exception as e:
            print(f"AudioCapture: probe failed: {e}")
            return False

    def _reader_loop(self):
        try:
            while self._running and self._process and self._process.poll() is None:
                data = self._process.stdout.read(self._chunk_bytes)
                if not data:
                    break
                t = time.time()
                with self._lock:
                    self._buffer.append((t, data))
        except Exception as e:
            print(f"AudioCapture: reader error: {e}")
        finally:
            self._cleanup_process()

    def _cleanup_process(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    def _write_wav(self, path: Path, pcm_data: bytes):
        import struct
        n = len(pcm_data)
        with open(path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + n))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))                          # PCM
            f.write(struct.pack("<H", self.channels))
            f.write(struct.pack("<I", self.sample_rate))
            f.write(struct.pack("<I", self.bytes_per_second))
            f.write(struct.pack("<H", self.channels * self.bytes_per_sample))
            f.write(struct.pack("<H", self.bytes_per_sample * 8))  # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", n))
            f.write(pcm_data)


# ---------------------------------------------------------------------------
# Clip transcoding / preview generation
# ---------------------------------------------------------------------------

def _transcode(src: Path, audio_path: Path | None = None):
    """Re-encode a clip to H.264, optionally muxing in audio."""
    tmp = src.with_suffix(".tmp.mp4")
    t0 = time.monotonic()

    if audio_path and audio_path.exists():
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(src),
             "-i", str(audio_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k",
             "-shortest",
             "-movflags", "+faststart",
             str(tmp)],
            capture_output=True,
        )
        audio_path.unlink(missing_ok=True)
    else:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", str(tmp)],
            capture_output=True,
        )

    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        tmp.replace(src)
        print(f"Transcode done in {elapsed:.1f}s: {src.name}")
        _generate_preview(src)
    else:
        tmp.unlink(missing_ok=True)
        if audio_path:
            audio_path.unlink(missing_ok=True)
        print(f"ffmpeg transcode failed for {src} ({elapsed:.1f}s): "
              f"{result.stderr.decode()[:300]}")


def _generate_preview(src: Path):
    """Generate a short looping preview MP4 (first 3 seconds, 320px wide)."""
    preview = src.with_name(src.stem + "_preview.mp4")
    if preview.exists():
        return
    t0 = time.monotonic()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-t", "3",
         "-vf", "scale=320:-2",
         "-c:v", "libx264", "-preset", "fast", "-crf", "28",
         "-c:a", "aac", "-b:a", "64k",
         "-movflags", "+faststart",
         str(preview)],
        capture_output=True,
    )
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"Preview done in {elapsed:.1f}s: {preview.name}")
    else:
        preview.unlink(missing_ok=True)
        print(f"Preview failed for {src.name}: {result.stderr.decode()[:200]}")


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
    "stop sign": 0,
    "parking meter": 0,
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
    "umbrella": 0,
    "handbag": 1,
    "tie": 0,
    "suitcase": 1,
    "frisbee": 0,
    "skis": 1,
    "snowboard": 1,
    "sports ball": 0,
    "kite": 0,
    "baseball bat": 0,
    "baseball glove": 0,
    "skateboard": 0,
    "surfboard": 0,
    "tennis racket": 0,
    "bottle": 0,
    "wine glass": 0,
    "cup": 0,
    "fork": 0,
    "knife": 0,
    "spoon": 0,
    "bowl": 0,
    "banana": 0,
    "apple": 0,
    "sandwich": 0,
    "orange": 0,
    "broccoli": 0,
    "carrot": 0,
    "hot dog": 0,
    "pizza": 0,
    "donut": 0,
    "cake": 0,
    "chair": 0,
    "couch": 0,
    "potted plant": 0,
    "bed": 0,
    "dining table": 0,
    "toilet": 0,
    "tv": 0,
    "laptop": 0,
    "mouse": 1,
    "remote": 0,
    "keyboard": 0,
    "cell phone": 0,
    "microwave": 0,
    "oven": 0,
    "toaster": 0,
    "sink": 0,
    "refrigerator": 0,
    "book": 0,
    "clock": 0,
    "vase": 0,
    "scissors": 0,
    "teddy bear": 0,
    "hair drier": 0,
    "toothbrush": 0,
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
# EventQueue — fan-out detection events to SSE handlers
# ---------------------------------------------------------------------------

class EventQueue:
    """Thread-safe fan-out queue: detector thread puts events, SSE handlers subscribe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: list[_queue.SimpleQueue] = []

    def subscribe(self) -> _queue.SimpleQueue:
        """Register a new SSE consumer; returns its private queue."""
        q: _queue.SimpleQueue = _queue.SimpleQueue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: _queue.SimpleQueue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def put(self, event: dict) -> None:
        """Broadcast an event to all current subscribers (called from detector thread)."""
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put(event)


# ---------------------------------------------------------------------------
# ClipRecorder
# ---------------------------------------------------------------------------

class ClipRecorder:
    """Records short video clips bracketing high-priority detections."""

    IDLE = "idle"
    RECORDING = "recording"

    def __init__(self, clips_dir: Path, fps: float, width: int, height: int,
                 pre_roll: float = 3.0, post_roll: float = 5.0, max_duration: float = 120.0,
                 audio_capture: AudioCapture | None = None):
        self.clips_dir = clips_dir
        self.fps = max(fps, 1.0)
        self.width = width
        self.height = height
        self.post_roll = post_roll
        self.max_duration = max_duration
        self._pre_roll = pre_roll
        self._audio: AudioCapture | None = audio_capture

        self.state = self.IDLE
        self._pre_buf: deque = deque(maxlen=max(1, int(self.fps * pre_roll)))
        self._writer: cv2.VideoWriter | None = None
        self._clip_id: str | None = None
        self._clip_path: Path | None = None
        self._start_time: float = 0.0
        self._last_det_time: float = 0.0
        self._best_conf: float = 0.0
        self._best_class: str | None = None
        self._wall_start: float = 0.0

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
        ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%dT%H-%M-%S")
        self._clip_id = f"{ts}_{class_name}"
        self._clip_path = self.clips_dir / f"{self._clip_id}.mp4"
        pre_roll_duration = len(self._pre_buf) / self.fps
        self._wall_start = time.time() - pre_roll_duration
        self._start_time = now
        self._last_det_time = now
        self._best_conf = conf
        self._best_class = class_name
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

        audio_path = None
        if self._audio is not None:
            audio_path = self._audio.extract(
                self._wall_start, time.time(), self._clip_path.parent,
            )

        threading.Thread(
            target=_transcode, args=(self._clip_path, audio_path), daemon=True,
        ).start()
        return {
            "id": self._clip_id,
            "path": str(self._clip_path),
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

def run(backend, frame_buffer: FrameBuffer, stop_event: threading.Event, args,
        reviews_path: Path, reviews_lock: threading.Lock, stats_monitor=None,
        event_queue: EventQueue | None = None):
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

    with backend:
        input_height = backend.input_height
        input_width = backend.input_width

        cap = open_source(source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_time = 1.0 / fps

        label = source_label(source)
        print(f"Source: {label} ({width}x{height} @ {fps:.2f} FPS)")

        # Start audio capture for RTSP sources
        audio_capture = None
        if kind == "rtsp":
            audio_capture = AudioCapture(source)
            if not audio_capture.start():
                audio_capture = None

        # Scale frame down if needed so tile count stays within the batch budget.
        # Only applicable when the backend has a fixed batch size (max_tiles is not None).
        if args.tile_overlap > 0 and backend.max_tiles is not None:
            proc_w, proc_h = _compute_downscale(
                width, height, input_height, args.tile_overlap, backend.max_tiles,
            )
            if proc_w != width or proc_h != height:
                print(f"Downscale: {width}x{height} → {proc_w}x{proc_h} "
                      f"to keep tiles ≤ {backend.max_tiles}")
        else:
            proc_w, proc_h = width, height

        if args.tile_overlap > 0:
            sample_tiles = make_tiles(
                np.zeros((proc_h, proc_w, 3), dtype=np.uint8),
                input_height, args.tile_overlap,
            )
            n_total = len(sample_tiles) + 1  # tiles + full-frame
            max_tiles_str = str(backend.max_tiles) if backend.max_tiles is not None else "unlimited"
            print(f"Tiling: {len(sample_tiles)} tiles + 1 full-frame = {n_total} inferences/frame "
                  f"({input_height}×{input_width}, {args.tile_overlap:.0%} overlap, max_tiles={max_tiles_str})")
        else:
            print(f"Single-pass: 1 inference/frame")

        clips_dir = Path(args.clips_dir)
        recorder = ClipRecorder(
            clips_dir, fps, proc_w, proc_h,
            pre_roll=args.pre_roll, post_roll=args.post_roll, max_duration=args.max_clip,
            audio_capture=audio_capture,
        )
        tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

        n_crop_tiles = len(sample_tiles) if args.tile_overlap > 0 else 0
        if stats_monitor is not None:
            stats_monitor.update({
                "stream_res": f"{proc_w}x{proc_h}",
                "n_tiles": n_crop_tiles,
            })

        _EMA_ALPHA = 0.1
        _ema_tile_fps: float | None = None
        _ema_frame_ms: float | None = None
        _ema_stream_fps: float | None = None
        _prev_loop_t: float | None = None

        try:
            while not stop_event.is_set():
                start = time.time()

                _t_now = time.monotonic()
                if _prev_loop_t is not None:
                    _dt = _t_now - _prev_loop_t
                    if _dt > 0:
                        _raw_sfps = 1.0 / _dt
                        _ema_stream_fps = (_EMA_ALPHA * _raw_sfps + (1 - _EMA_ALPHA) * _ema_stream_fps
                                           if _ema_stream_fps is not None else _raw_sfps)
                _prev_loop_t = _t_now

                ret, frame = cap.read()
                if not ret:
                    if kind in ("usb", "rtsp"):
                        if kind == "rtsp":
                            print("RTSP: lost connection, reconnecting...")
                            cap.release()
                            time.sleep(2)
                            cap = open_source(source)
                            if audio_capture is not None:
                                audio_capture.restart()
                        continue
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                if proc_w != width or proc_h != height:
                    frame = cv2.resize(frame, (proc_w, proc_h))

                tiles = make_tiles(frame, input_height, args.tile_overlap) if args.tile_overlap > 0 else []
                tile_imgs = [t[0] for t in tiles]

                t0 = time.monotonic()
                per_image_dets = backend.infer(tile_imgs + [frame])
                t_elapsed = time.monotonic() - t0

                # Collect tile detections (translate from tile-space to frame-space)
                all_dets = []
                for i, (_, tx, ty) in enumerate(tiles):
                    for d in per_image_dets[i]:
                        d = list(d)
                        d[0] += tx; d[2] += tx
                        d[1] += ty; d[3] += ty
                        all_dets.append(d)

                # Full-frame detections (last element)
                all_dets.extend(per_image_dets[len(tiles)])

                detections = nms(all_dets)

                if t_elapsed > 0 and stats_monitor is not None:
                    n_infer = len(tiles) + 1
                    raw_tile_fps = n_infer / t_elapsed
                    raw_frame_ms = t_elapsed * 1000
                    if _ema_tile_fps is None:
                        _ema_tile_fps = raw_tile_fps
                        _ema_frame_ms = raw_frame_ms
                    else:
                        _ema_tile_fps = _EMA_ALPHA * raw_tile_fps + (1 - _EMA_ALPHA) * _ema_tile_fps
                        _ema_frame_ms = _EMA_ALPHA * raw_frame_ms + (1 - _EMA_ALPHA) * _ema_frame_ms
                    stats_monitor.update({
                        **backend.get_hw_stats(),
                        "tile_fps": round(_ema_tile_fps, 1),
                        "frame_ms": round(_ema_frame_ms, 1),
                        "stream_fps": round(_ema_stream_fps, 1) if _ema_stream_fps is not None else None,
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
                    if event_queue is not None:
                        event_queue.put(clip_info)

                elapsed = time.time() - start
                if kind != "rtsp" and elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        finally:
            clip_info = recorder.force_close()
            if clip_info is not None:
                reviews_mod.add(reviews_path, reviews_lock, clip_info)
                if event_queue is not None:
                    event_queue.put(clip_info)
            cap.release()
            if audio_capture is not None:
                audio_capture.stop()
            csv_file.close()
