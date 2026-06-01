#!/usr/bin/env python3
"""Hardware PWM fan controller for deer deterrent.

Uses the Linux sysfs PWM interface (/sys/class/pwm/) for jitter-free
hardware PWM on a Raspberry Pi.  Requires the pwm device-tree overlay
to be enabled in /boot/firmware/config.txt:

    dtoverlay=pwm,pin=18,func=2

A reboot is needed after adding the overlay.

Usage:
    python3 fan.py                          # defaults: 50 Hz, 5% duty, port 8080
    python3 fan.py --freq 1000 --duty 50    # 1 kHz, 50%
    python3 fan.py --port 9000              # custom HTTP port

HTTP API:
    GET  /              → current status (JSON)
    GET  /metrics       → Prometheus metrics
    POST /duty?value=75 → set duty cycle to 75%
"""

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PWM_BASE = Path("/sys/class/pwm")
MAX_TRIGGER_HISTORY = 100


def export_channel(chip: int, channel: int) -> Path:
    chip_path = PWM_BASE / f"pwmchip{chip}"
    if not chip_path.exists():
        sys.exit(
            f"Error: {chip_path} not found. "
            "Ensure the pwm overlay is enabled in /boot/firmware/config.txt "
            "and reboot."
        )
    chan_path = chip_path / f"pwm{channel}"
    if not chan_path.exists():
        (chip_path / "export").write_text(str(channel))
        for _ in range(20):
            if chan_path.exists():
                break
            time.sleep(0.1)
        else:
            sys.exit(f"Error: {chan_path} did not appear after export.")
    return chan_path


class FanController:
    def __init__(self, chan: Path, freq_hz: int, duty_pct: float):
        self.chan = chan
        self.freq_hz = freq_hz
        self.period_ns = int(1_000_000_000 / freq_hz)
        self.max_duty = 8
        self.running = False
        self.start_time = time.time()
        self.trigger_count = 0
        self.duty_changes = deque(maxlen=MAX_TRIGGER_HISTORY)

        if duty_pct is None:
            self.duty_pct = 0
        else:
            self.duty_pct = min(duty_pct, self.max_duty)

        duty_ns = int(self.period_ns * self.duty_pct / 100.0)
        (self.chan / "enable").write_text("0")
        (self.chan / "period").write_text(str(self.period_ns))
        (self.chan / "duty_cycle").write_text(str(duty_ns))
        (self.chan / "enable").write_text("1")
        self.running = True
        print(f"PWM: {self.freq_hz} Hz, {self.duty_pct}% duty")

    def set_duty(self, duty_pct: float):
        old = self.duty_pct
        self.duty_pct = max(0.0, min(100.0, duty_pct))
        if self.running:
            duty_ns = int(self.period_ns * self.duty_pct / 100.0)
            (self.chan / "duty_cycle").write_text(str(duty_ns))
        # Track non-zero transitions as triggers
        if old == 0 and self.duty_pct > 0:
            self.trigger_count += 1
        self.duty_changes.append((time.time(), old, self.duty_pct))
        print(f"PWM: duty → {self.duty_pct}%")

    def stop(self):
        try:
            (self.chan / "enable").write_text("0")
            (self.chan / "duty_cycle").write_text("0")
        except OSError:
            pass
        self.running = False
        print("PWM: stopped")

    def status(self) -> dict:
        return {
            "running": self.running,
            "freq_hz": self.freq_hz,
            "duty_pct": self.duty_pct,
            "uptime_secs": int(time.time() - self.start_time),
            "trigger_count": self.trigger_count,
        }

    def prometheus_metrics(self) -> str:
        uptime = time.time() - self.start_time
        load1, load5, load15 = os.getloadavg()
        try:
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    parts = line.split()
                    if parts[0] in ("MemTotal:", "MemAvailable:"):
                        mem[parts[0].rstrip(":")] = int(parts[1]) * 1024
        except Exception:
            mem = {}

        # WiFi signal strength
        wifi_signal = -1
        try:
            with open("/proc/net/wireless") as f:
                for line in f:
                    if "wlan0" in line:
                        wifi_signal = int(float(line.split()[3]))
        except Exception:
            pass

        # CPU temperature
        cpu_temp = 0
        try:
            cpu_temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
        except Exception:
            pass

        lines = [
            f"# HELP fan_running Whether the fan PWM is enabled",
            f"# TYPE fan_running gauge",
            f"fan_running {1 if self.running else 0}",
            f"# HELP fan_duty_pct Current duty cycle percentage",
            f"# TYPE fan_duty_pct gauge",
            f"fan_duty_pct {self.duty_pct}",
            f"# HELP fan_freq_hz PWM frequency",
            f"# TYPE fan_freq_hz gauge",
            f"fan_freq_hz {self.freq_hz}",
            f"# HELP fan_trigger_count_total Times fan was activated from 0",
            f"# TYPE fan_trigger_count_total counter",
            f"fan_trigger_count_total {self.trigger_count}",
            f"# HELP fan_uptime_seconds Seconds since fan.py started",
            f"# TYPE fan_uptime_seconds gauge",
            f"fan_uptime_seconds {uptime:.0f}",
            f"# HELP node_load1 1-minute load average",
            f"# TYPE node_load1 gauge",
            f"node_load1 {load1:.2f}",
            f"# HELP node_load5 5-minute load average",
            f"# TYPE node_load5 gauge",
            f"node_load5 {load5:.2f}",
            f"# HELP node_memory_total_bytes Total memory",
            f"# TYPE node_memory_total_bytes gauge",
            f'node_memory_total_bytes {mem.get("MemTotal", 0)}',
            f"# HELP node_memory_available_bytes Available memory",
            f"# TYPE node_memory_available_bytes gauge",
            f'node_memory_available_bytes {mem.get("MemAvailable", 0)}',
            f"# HELP node_wifi_signal_dbm WiFi signal strength",
            f"# TYPE node_wifi_signal_dbm gauge",
            f"node_wifi_signal_dbm {wifi_signal}",
            f"# HELP node_cpu_temp_celsius CPU temperature",
            f"# TYPE node_cpu_temp_celsius gauge",
            f"node_cpu_temp_celsius {cpu_temp:.1f}",
        ]
        return "\n".join(lines) + "\n"


def make_handler(fan: FanController):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._json(200, fan.status())
            elif path == "/metrics":
                body = fan.prometheus_metrics().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/duty":
                raw = params.get("value", [None])[0]
                if raw is None:
                    self._json(400, {"error": "missing ?value= parameter"})
                    return
                try:
                    value = float(raw)
                except ValueError:
                    self._json(400, {"error": "value must be a number"})
                    return
                if not (0 <= value <= 100):
                    self._json(400, {"error": "value must be 0-100"})
                    return
                fan.set_duty(value)
                self._json(200, fan.status())
            else:
                self._json(404, {"error": "not found"})

        def _json(self, code: int, data: dict):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass  # silence per-request logs

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Hardware PWM fan controller")
    parser.add_argument("--freq", type=int, default=50,
                        help="PWM frequency in Hz (default: 50)")
    parser.add_argument("--duty", type=float, default=5.0,
                        help="Duty cycle 0-100 (default: 5)")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP server port (default: 8080)")
    parser.add_argument("--pwm-chip", type=int, default=0,
                        help="PWM chip number (default: 0)")
    parser.add_argument("--pwm-channel", type=int, default=0,
                        help="PWM channel number (default: 0)")
    args = parser.parse_args()

    if not (0 <= args.duty <= 100):
        sys.exit("Error: --duty must be between 0 and 100")
    if args.freq <= 0:
        sys.exit("Error: --freq must be positive")

    chan = export_channel(args.pwm_chip, args.pwm_channel)
    fan = FanController(chan, args.freq, args.duty)

    def shutdown(sig, frame):
        fan.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server = HTTPServer(("0.0.0.0", args.port), make_handler(fan))
    print(f"HTTP server on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
