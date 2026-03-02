#!/usr/bin/env python3
"""Backfill preview MP4s for all existing clips that don't have one yet."""

import subprocess
import sys
from pathlib import Path

clips_dir = Path(__file__).parent / "clips"

if not clips_dir.exists():
    print(f"Clips directory not found: {clips_dir}")
    sys.exit(1)

clips = sorted(clips_dir.glob("*.mp4"))
# Exclude files that are already previews
clips = [c for c in clips if not c.stem.endswith("_preview")]

total = len(clips)
generated = 0
skipped = 0

for i, clip in enumerate(clips, 1):
    preview = clip.with_name(clip.stem + "_preview.mp4")
    if preview.exists():
        skipped += 1
        continue

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(clip),
         "-t", "3",
         "-vf", "scale=320:-2",
         "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "28",
         "-movflags", "+faststart",
         str(preview)],
        capture_output=True,
    )
    if result.returncode == 0:
        generated += 1
        print(f"[{i}/{total}] {preview.name}")
    else:
        print(f"[{i}/{total}] FAILED: {clip.name}", file=sys.stderr)

print(f"\nDone: {generated} generated, {skipped} skipped, {total} total")
