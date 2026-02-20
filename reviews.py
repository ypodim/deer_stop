"""Helpers for reading and writing reviews.json."""

import json
import threading
from pathlib import Path


def load(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text())
    return []


def save(path: Path, reviews: list):
    path.write_text(json.dumps(reviews, indent=2))


def add(path: Path, lock: threading.Lock, clip_info: dict):
    """Thread-safe append of a new clip entry, then persist."""
    with lock:
        reviews = load(path)
        reviews.append(clip_info)
        save(path, reviews)
    print(f"Clip saved: {clip_info['path']}")


def prune(path: Path, lock: threading.Lock):
    """Remove entries whose clip files no longer exist on disk."""
    if not path.exists():
        return
    with lock:
        reviews = load(path)
        kept = [r for r in reviews if Path(r["path"]).exists()]
        removed = len(reviews) - len(kept)
        if removed:
            save(path, kept)
            print(f"Pruned {removed} stale review entr{'y' if removed == 1 else 'ies'}")
