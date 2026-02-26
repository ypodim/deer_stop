"""Hailo + CPU stats sampler.

Runs hailortcli monitor in a background thread to read Hailo device
utilization and FPS. Requires HAILO_MONITOR=1 in the process environment
before the Hailo VDevice is opened.
"""

import re
import subprocess
import threading
import time

import psutil

# Strip all ANSI/VT100 escape sequences
_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _parse_monitor(text: str) -> dict:
    """Parse hailortcli monitor output into a stats dict."""
    result = {
        "hailo_device_util": None,
        "hailo_model_util":  None,
        "hailo_fps":         None,
    }
    section = None
    past_sep = False

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        if "Device ID" in line and "Utilization" in line and "Architecture" in line:
            section = "device"
            past_sep = False
        elif "Model" in line and "Utilization" in line and "FPS" in line and "PID" in line:
            section = "model"
            past_sep = False
        elif re.match(r"^-{10,}", s):
            past_sep = True
        elif past_sep and section == "device":
            # Strip PCI address (e.g. "0001:01:00.0") before reading the utilization
            stripped = re.sub(r'\b[0-9a-fA-F]+(?::[0-9a-fA-F]+)+(?:\.[0-9a-fA-F]+)?\b', '', line)
            nums = re.findall(r"\d+(?:\.\d+)?", stripped)
            if nums:
                result["hailo_device_util"] = float(nums[0])
        elif past_sep and section == "model":
            nums = re.findall(r"\d+(?:\.\d+)?", line)
            if len(nums) >= 2:
                result["hailo_model_util"] = float(nums[0])
                result["hailo_fps"] = float(nums[1])

    return result


class StatsMonitor:
    """Samples Hailo and CPU stats in a background daemon thread.

    Call get() from any thread to retrieve the latest snapshot.
    """

    def __init__(self, interval: float = 2.0):
        self._interval = interval
        self._lock = threading.Lock()
        self._stats: dict = {
            "hailo_device_util": None,
            "hailo_model_util":  None,
            "hailo_fps":         None,
            "cpu_percent":       None,
            "tile_fps":          None,
            "frame_ms":          None,
        }
        # Prime psutil so the first real reading is non-zero
        psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._run, daemon=True, name="StatsMonitor")
        self._thread.start()

    def get(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def update(self, extra: dict):
        """Merge additional stats (e.g. from the detector thread)."""
        with self._lock:
            self._stats.update(extra)

    def _run(self):
        while True:
            cpu = psutil.cpu_percent(interval=None)
            hailo = self._poll_hailo()
            with self._lock:
                self._stats.update({"cpu_percent": cpu, **hailo})
            time.sleep(self._interval)

    def _poll_hailo(self) -> dict:
        try:
            proc = subprocess.run(
                ["hailortcli", "monitor"],
                capture_output=True,
                timeout=self._interval * 0.8,
            )
            raw = proc.stdout
        except subprocess.TimeoutExpired as e:
            raw = e.stdout or b""
        except FileNotFoundError:
            return {}

        text = _ANSI.sub("", raw.decode("utf-8", errors="replace"))
        return _parse_monitor(text)
