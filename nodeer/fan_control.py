#!/usr/bin/env python3
"""Local keyboard controller for the nodeer fan.

Use arrow keys to adjust duty cycle, sent via HTTP to the fan service.

Controls:
    ↑ / ↓   — adjust duty cycle by step
    0       — stop (duty 0)
    m       — max (duty 100)
    q / Esc — quit
"""

import sys
import tty
import termios
import urllib.request

HOST = "http://nodeer:8080"
STEP = 5


def post(path: str) -> dict | None:
    try:
        req = urllib.request.Request(f"{HOST}{path}", method="POST")
        with urllib.request.urlopen(req, timeout=2) as resp:
            import json
            return json.loads(resp.read())
    except Exception as e:
        print(f"\r\x1b[K  Error: {e}", end="", flush=True)
        return None


def get_status() -> dict | None:
    try:
        with urllib.request.urlopen(f"{HOST}/", timeout=2) as resp:
            import json
            return json.loads(resp.read())
    except Exception:
        return None


def draw(duty: float, running: bool):
    bar_len = 40
    filled = int(bar_len * duty / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    state = "ON " if running else "OFF"
    print(f"\r\x1b[K  [{bar}] {duty:5.1f}%  {state}  (↑/↓ step, 0=stop, m=max, q=quit)", end="", flush=True)


def read_key() -> str:
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        seq = sys.stdin.read(2)
        if seq == "[A":
            return "up"
        elif seq == "[B":
            return "down"
        return "esc"
    return ch


def main():
    status = get_status()
    if status is None:
        print(f"Cannot reach {HOST} — is fan.py running on nodeer?")
        sys.exit(1)

    duty = status["duty_pct"]
    running = status["running"]

    print("Fan Control — nodeer")
    print(f"Connected. Current: {duty}% {'(running)' if running else '(stopped)'}")
    print()
    draw(duty, running)

    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin)
        while True:
            key = read_key()
            if key in ("q", "\x03", "esc"):  # q, Ctrl-C, Esc
                break
            elif key == "up":
                duty = min(100, duty + STEP)
                result = post(f"/duty?value={duty}")
                if result:
                    duty = result["duty_pct"]
                    running = result["running"]
            elif key == "down":
                duty = max(0, duty - STEP)
                result = post(f"/duty?value={duty}")
                if result:
                    duty = result["duty_pct"]
                    running = result["running"]
            elif key == "0":
                result = post("/stop")
                if result:
                    duty = result["duty_pct"]
                    running = False
            elif key == "m":
                result = post("/duty?value=100")
                if result:
                    duty = result["duty_pct"]
                    running = result["running"]
            else:
                continue
            draw(duty, running)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print()


if __name__ == "__main__":
    main()
