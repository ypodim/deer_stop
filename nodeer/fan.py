#!/usr/bin/env python3
"""Hardware PWM fan controller for deer deterrent.

Uses the Linux sysfs PWM interface (/sys/class/pwm/) for jitter-free
hardware PWM on a Raspberry Pi.  Requires the pwm device-tree overlay
to be enabled in /boot/firmware/config.txt:

    dtoverlay=pwm,pin=18,func=2

A reboot is needed after adding the overlay.

Usage:
    python3 fan.py                          # defaults: 25 kHz, 100% duty, port 8080
    python3 fan.py --freq 1000 --duty 50    # 1 kHz, 50%
    python3 fan.py --port 9000              # custom HTTP port

HTTP API:
    GET  /              → current status (JSON)
    POST /duty?value=75 → set duty cycle to 75%
    POST /stop          → stop PWM (duty 0, disabled)
    POST /start         → restart PWM at last duty cycle
"""

import argparse
import json
import signal
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PWM_BASE = Path("/sys/class/pwm")


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
        self.duty_pct = duty_pct
        self.running = False

    def start(self, duty_pct: float | None = None):
        if duty_pct is not None:
            self.duty_pct = duty_pct
        duty_ns = int(self.period_ns * self.duty_pct / 100.0)
        (self.chan / "enable").write_text("0")
        (self.chan / "period").write_text(str(self.period_ns))
        (self.chan / "duty_cycle").write_text(str(duty_ns))
        (self.chan / "enable").write_text("1")
        self.running = True
        print(f"PWM: {self.freq_hz} Hz, {self.duty_pct}% duty")

    def set_duty(self, duty_pct: float):
        self.duty_pct = max(0.0, min(100.0, duty_pct))
        if self.running:
            duty_ns = int(self.period_ns * self.duty_pct / 100.0)
            (self.chan / "duty_cycle").write_text(str(duty_ns))
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
        }


def make_handler(fan: FanController):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlparse(self.path).path == "/":
                self._json(200, fan.status())
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
            elif path == "/stop":
                fan.stop()
                self._json(200, fan.status())
            elif path == "/start":
                fan.start()
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
    parser.add_argument("--freq", type=int, default=25000,
                        help="PWM frequency in Hz (default: 25000)")
    parser.add_argument("--duty", type=float, default=100.0,
                        help="Duty cycle 0-100 (default: 100)")
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
    fan.start()

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
