#!/usr/bin/env python3
"""Hardware PWM fan controller for deer deterrent.

Uses the Linux sysfs PWM interface (/sys/class/pwm/) for jitter-free
hardware PWM on a Raspberry Pi.  Requires the pwm device-tree overlay
to be enabled in /boot/firmware/config.txt:

    dtoverlay=pwm,pin=18,func=2

A reboot is needed after adding the overlay.

Usage:
    python3 fan.py                  # defaults: GPIO 18, 25 kHz, 100% duty
    python3 fan.py --freq 1000      # 1 kHz
    python3 fan.py --duty 50        # 50% duty cycle
    python3 fan.py --pwm-chip 0 --pwm-channel 0
"""

import argparse
import signal
import sys
import time
from pathlib import Path

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
        # Wait for sysfs to create the channel
        for _ in range(20):
            if chan_path.exists():
                break
            time.sleep(0.1)
        else:
            sys.exit(f"Error: {chan_path} did not appear after export.")
    return chan_path


def start_pwm(chan: Path, freq_hz: int, duty_pct: float):
    period_ns = int(1_000_000_000 / freq_hz)
    duty_ns = int(period_ns * duty_pct / 100.0)

    # Must disable before changing parameters
    (chan / "enable").write_text("0")
    (chan / "period").write_text(str(period_ns))
    (chan / "duty_cycle").write_text(str(duty_ns))
    (chan / "enable").write_text("1")

    print(f"PWM started: {freq_hz} Hz, {duty_pct}% duty cycle "
          f"(period={period_ns} ns, duty={duty_ns} ns)")


def stop_pwm(chan: Path):
    try:
        (chan / "enable").write_text("0")
        (chan / "duty_cycle").write_text("0")
    except OSError:
        pass
    print("PWM stopped.")


def main():
    parser = argparse.ArgumentParser(description="Hardware PWM fan controller")
    parser.add_argument("--freq", type=int, default=25000,
                        help="PWM frequency in Hz (default: 25000)")
    parser.add_argument("--duty", type=float, default=100.0,
                        help="Duty cycle 0-100 (default: 100)")
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
    start_pwm(chan, args.freq, args.duty)

    def shutdown(sig, frame):
        stop_pwm(chan)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Running (Ctrl+C to stop)...")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
