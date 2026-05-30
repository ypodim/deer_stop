"""Detection notifications: email alert + fan trigger.

Deduplicates by class name — won't send consecutive emails for the
same animal.  Resets after COOLDOWN_SECS of no detections.
"""

import subprocess
import threading
import time
import urllib.request

COOLDOWN_SECS = 300  # 5 min — reset "last class" after this gap
FAN_URL = "http://nodeer:8080/duty?value=7"

_last_class: str | None = None
_last_time: float = 0
_lock = threading.Lock()


def _send_email(subject: str, body: str, to: str):
    """Send email via msmtp (configured in ~/.msmtprc or /etc/msmtprc)."""
    message = f"From: deerstop@astrapi\nTo: {to}\nSubject: {subject}\n\n{body}"
    try:
        proc = subprocess.run(
            ["msmtp", "--", to],
            input=message, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            print(f"Notify: emailed {to} — {subject}")
        else:
            print(f"Notify: email failed — {proc.stderr.strip()}")
    except Exception as e:
        print(f"Notify: email failed — {e}")


def _trigger_fan():
    """Send HTTP request to nodeer fan controller."""
    try:
        req = urllib.request.Request(FAN_URL, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"Notify: fan triggered — {resp.read().decode()}")
    except Exception as e:
        print(f"Notify: fan trigger failed — {e}")


def on_clip(clip_info: dict, email_to: str):
    """Called when a clip is saved.  Runs in a background thread."""
    global _last_class, _last_time

    class_name = clip_info.get("class_name", "unknown")
    now = time.time()

    with _lock:
        # Reset if cooldown elapsed
        if now - _last_time > COOLDOWN_SECS:
            _last_class = None

        # Skip if same animal as last notification
        if class_name == _last_class:
            print(f"Notify: skipping duplicate {class_name}")
            return

        _last_class = class_name
        _last_time = now

    conf = clip_info.get("confidence", 0)
    ts = clip_info.get("timestamp", "")

    threading.Thread(target=_send_email, args=(
        f"DeerStop: {class_name} detected",
        f"{class_name} detected at {ts} (confidence: {conf:.0%})",
        email_to,
    ), daemon=True).start()

    threading.Thread(target=_trigger_fan, daemon=True).start()
