"""Backend-agnostic stats store shared between the detector thread and the web UI."""

import threading
import time

import psutil


class StatsStore:
    """Thread-safe dict store.

    The detector backend calls update() to publish metrics.
    The web UI calls get() to read the latest snapshot.
    CPU utilization is polled here so it is available regardless of backend.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict = {}
        psutil.cpu_percent(interval=None)  # prime so first real reading is non-zero
        t = threading.Thread(target=self._poll_cpu, daemon=True, name="CpuPoller")
        t.start()

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def update(self, extra: dict):
        with self._lock:
            self._data.update(extra)

    def _poll_cpu(self):
        while True:
            cpu = psutil.cpu_percent(interval=None)
            with self._lock:
                self._data["cpu_percent"] = cpu
            time.sleep(2.0)
