"""Detection notifications: email alert + fan trigger.

Two entry points:
  on_detection() — called immediately when a high-priority frame is detected.
                   Triggers fan instantly. Logs detection timestamp.
  on_clip()      — called when a clip finishes recording.
                   Sends email notification.

Fan logic: on first detection, turn fan to 7% for 10 seconds then back to 5%.
Won't re-trigger for FAN_COOLDOWN_SECS.

Timing metrics exposed via get_metrics() for Prometheus scraping.
"""

import subprocess
import threading
import time
import urllib.request

EMAIL_COOLDOWN_SECS = 300   # 5 min — reset "last class" after this gap
FAN_COOLDOWN_SECS = 300     # 5 min — don't re-trigger fan within this window
FAN_TRIGGER_URL = "http://nodeer:8080/trigger"

_last_class: str | None = None
_last_email_time: float = 0
_last_fan_time: float = 0
_lock = threading.Lock()

# Timing metrics
_last_detection_time: float = 0      # when YOLO detected the animal
_last_fan_request_time: float = 0    # when HTTP request was sent
_last_fan_response_time: float = 0   # when HTTP response came back
_last_email_send_time: float = 0     # when email was sent
_last_clip_close_time: float = 0     # when clip finished recording
_detection_to_fan_ms: float = 0      # detection → fan activated
_detection_to_email_ms: float = 0    # detection → email sent
_detection_to_clip_ms: float = 0     # detection → clip closed
_fan_http_ms: float = 0              # fan HTTP round trip


def _send_email(subject: str, body: str, to: str):
    """Send email via msmtp."""
    global _last_email_send_time, _detection_to_email_ms
    message = f"From: deerstop@astrapi\nTo: {to}\nSubject: {subject}\n\n{body}"
    try:
        proc = subprocess.run(
            ["msmtp", "--", to],
            input=message, capture_output=True, text=True, timeout=30,
        )
        _last_email_send_time = time.time()
        if _last_detection_time > 0:
            _detection_to_email_ms = (_last_email_send_time - _last_detection_time) * 1000
        if proc.returncode == 0:
            print(f"Notify: emailed {to} — {subject} ({_detection_to_email_ms:.0f}ms from detection)")
        else:
            print(f"Notify: email failed — {proc.stderr.strip()}")
    except Exception as e:
        print(f"Notify: email failed — {e}")


def _fan_trigger():
    """Send trigger request to fan service (handles on/off cycle server-side)."""
    global _last_fan_request_time, _last_fan_response_time, _fan_http_ms, _detection_to_fan_ms
    t0 = _last_detection_time
    _last_fan_request_time = time.time()
    try:
        req = urllib.request.Request(FAN_TRIGGER_URL, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            _last_fan_response_time = time.time()
            _fan_http_ms = (_last_fan_response_time - _last_fan_request_time) * 1000
            if t0 > 0:
                _detection_to_fan_ms = (_last_fan_response_time - t0) * 1000
                print(f"Notify: fan triggered (HTTP {_fan_http_ms:.0f}ms, detection→fan {_detection_to_fan_ms:.0f}ms)")
            else:
                print(f"Notify: fan triggered (HTTP {_fan_http_ms:.0f}ms)")
    except Exception as e:
        _last_fan_response_time = time.time()
        _fan_http_ms = (_last_fan_response_time - _last_fan_request_time) * 1000
        print(f"Notify: fan trigger failed — {e} ({_fan_http_ms:.0f}ms)")


def on_detection(class_name: str, conf: float, notify_email: str, camera: str = ""):
    """Called immediately when a high-priority detection fires.
    Triggers fan instantly, no waiting for clip to close.

    The fan cooldown is shared across all cameras (there is only one fan)."""
    global _last_detection_time, _last_fan_time

    _last_detection_time = time.time()
    now = _last_detection_time
    tag = f"[{camera}] " if camera else ""

    with _lock:
        trigger_fan = now - _last_fan_time >= FAN_COOLDOWN_SECS
        if trigger_fan:
            _last_fan_time = now

    if trigger_fan:
        print(f"Notify: {tag}{class_name} detected ({conf:.0%}), triggering fan")
        threading.Thread(target=_fan_trigger, daemon=True).start()
    else:
        remaining = int(FAN_COOLDOWN_SECS - (now - _last_fan_time))
        print(f"Notify: {tag}{class_name} detected, fan cooldown ({remaining}s remaining)")


def on_clip(clip_info: dict, email_to: str):
    """Called when a clip finishes recording. Sends email."""
    global _last_class, _last_email_time, _last_clip_close_time, _detection_to_clip_ms

    class_name = clip_info.get("class_name", "unknown")
    camera = clip_info.get("camera", "")
    now = time.time()
    _last_clip_close_time = now
    # Dedup key is per (camera, class) so different cameras still notify.
    dedup_key = f"{camera}:{class_name}"

    if _last_detection_time > 0:
        _detection_to_clip_ms = (now - _last_detection_time) * 1000
        print(f"Notify: clip closed, {_detection_to_clip_ms:.0f}ms since last detection")

    with _lock:
        if now - _last_email_time > EMAIL_COOLDOWN_SECS:
            _last_class = None

        send_email = dedup_key != _last_class
        if send_email:
            _last_class = dedup_key
            _last_email_time = now

    if send_email:
        conf = clip_info.get("confidence", 0)
        ts_raw = clip_info.get("timestamp", "")
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            dt = datetime.fromisoformat(ts_raw).astimezone(ZoneInfo("America/Los_Angeles"))
            ts = dt.strftime("%Y-%m-%d %I:%M:%S %p PT")
        except Exception:
            ts = ts_raw
        where = f" on {camera}" if camera else ""
        threading.Thread(target=_send_email, args=(
            f"DeerStop: {class_name} detected{where}",
            f"{class_name} detected{where} at {ts} (confidence: {conf:.0%})",
            email_to,
        ), daemon=True).start()
    else:
        print(f"Notify: skipping duplicate email for {dedup_key}")


def get_metrics() -> str:
    """Prometheus metrics for the notification pipeline timing."""
    lines = [
        "# HELP deerstop_detection_to_fan_ms Delay from detection to fan activation",
        "# TYPE deerstop_detection_to_fan_ms gauge",
        f"deerstop_detection_to_fan_ms {_detection_to_fan_ms:.1f}",
        "# HELP deerstop_detection_to_clip_ms Delay from last detection to clip close",
        "# TYPE deerstop_detection_to_clip_ms gauge",
        f"deerstop_detection_to_clip_ms {_detection_to_clip_ms:.1f}",
        "# HELP deerstop_detection_to_email_ms Delay from detection to email sent",
        "# TYPE deerstop_detection_to_email_ms gauge",
        f"deerstop_detection_to_email_ms {_detection_to_email_ms:.1f}",
        "# HELP deerstop_fan_http_ms Fan HTTP round trip time",
        "# TYPE deerstop_fan_http_ms gauge",
        f"deerstop_fan_http_ms {_fan_http_ms:.1f}",
    ]
    return "\n".join(lines) + "\n"
