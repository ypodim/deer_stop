"""Detection notifications: email alert + fan trigger.

Deduplicates by class name — won't send consecutive emails for the
same animal.  Resets after EMAIL_COOLDOWN_SECS of no detections.

Fan logic: on detection, turn fan on at 7% for 10 seconds then off.
Won't trigger again for FAN_COOLDOWN_SECS (10 min).
"""

import subprocess
import threading
import time
import urllib.request

EMAIL_COOLDOWN_SECS = 300   # 5 min — reset "last class" after this gap
FAN_COOLDOWN_SECS = 600     # 10 min — don't re-trigger fan within this window
FAN_ON_SECS = 10            # how long the fan stays on
FAN_DUTY = 7
FAN_URL = "http://nodeer:8080/duty?value="

_last_class: str | None = None
_last_email_time: float = 0
_last_fan_time: float = 0
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


def _fan_set(duty: float):
    """Set fan duty cycle via HTTP."""
    try:
        req = urllib.request.Request(f"{FAN_URL}{duty}", method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"Notify: fan → {duty}% — {resp.read().decode()}")
    except Exception as e:
        print(f"Notify: fan request failed — {e}")


def _fan_on_then_off():
    """Turn fan on, wait, turn off."""
    _fan_set(FAN_DUTY)
    time.sleep(FAN_ON_SECS)
    _fan_set(0)


def on_clip(clip_info: dict, email_to: str):
    """Called when a clip is saved."""
    global _last_class, _last_email_time, _last_fan_time

    class_name = clip_info.get("class_name", "unknown")
    now = time.time()

    with _lock:
        # --- Email dedup ---
        if now - _last_email_time > EMAIL_COOLDOWN_SECS:
            _last_class = None

        send_email = class_name != _last_class
        if send_email:
            _last_class = class_name
            _last_email_time = now

        # --- Fan cooldown ---
        trigger_fan = now - _last_fan_time >= FAN_COOLDOWN_SECS
        if trigger_fan:
            _last_fan_time = now

    if send_email:
        conf = clip_info.get("confidence", 0)
        ts = clip_info.get("timestamp", "")
        threading.Thread(target=_send_email, args=(
            f"DeerStop: {class_name} detected",
            f"{class_name} detected at {ts} (confidence: {conf:.0%})",
            email_to,
        ), daemon=True).start()
    else:
        print(f"Notify: skipping duplicate email for {class_name}")

    if trigger_fan:
        threading.Thread(target=_fan_on_then_off, daemon=True).start()
    else:
        remaining = int(FAN_COOLDOWN_SECS - (now - _last_fan_time))
        print(f"Notify: fan cooldown, {remaining}s remaining")
