#!/usr/bin/env python3
"""
Pipeline latency profiler for the Hailo inference path.

Measures per-stage wall-clock time across N frames and reports mean/p50/p95/p99
for each stage, plus derived throughput figures.

Usage:
    python3 bench.py [--frames N] [--warmup N] [--source PATH_OR_URL]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from time import perf_counter

import numpy as np

os.environ["HAILO_MONITOR"] = "1"

from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams,
    FormatType,
)

# Reuse project helpers
sys.path.insert(0, str(Path(__file__).parent))
from detector import make_tiles, preprocess, postprocess, nms, COCO_NAMES, COCO_CLASSES
import monitor as mon

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HEF_PATH    = Path(__file__).parent / "yolov11l_b8.hef"
SOURCE      = Path(__file__).parent / "deer.mp4"
BATCH_SIZE  = 8
TILE_OVERLAP = 0.5
CONF        = 0.6
WARMUP      = 10   # frames to discard before recording


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def pct(arr, p):
    return float(np.percentile(arr, p))

def report(label, arr_ms, width=14):
    a = np.array(arr_ms)
    print(f"  {label:<{width}}  mean={a.mean():7.2f}  p50={pct(a,50):7.2f}"
          f"  p95={pct(a,95):7.2f}  p99={pct(a,99):7.2f}  ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames",  type=int, default=100)
    ap.add_argument("--warmup",  type=int, default=WARMUP)
    ap.add_argument("--source",  default=str(SOURCE))
    args = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {args.source}")
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Source:  {args.source}  ({src_w}×{src_h} @ {src_fps:.1f} FPS)")

    print(f"Loading: {HEF_PATH}")
    hef = HEF(str(HEF_PATH))
    input_info  = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    input_h, input_w = input_info.shape[0], input_info.shape[1]
    input_name  = input_info.name
    output_name = output_info.name
    print(f"Model:   {input_w}×{input_h}  batch={BATCH_SIZE}")

    # Count tiles on a sample frame
    sample = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    sample_tiles = make_tiles(sample, input_h, TILE_OVERLAP)
    n_tiles = len(sample_tiles)
    n_infer = n_tiles + 1  # tiles + full-frame
    input_bytes = BATCH_SIZE * input_h * input_w * 3
    print(f"Tiling:  {n_tiles} tiles + 1 full-frame = {n_infer} inferences/frame  "
          f"(batch padded to {BATCH_SIZE})")
    print(f"Input:   {input_bytes/1024/1024:.1f} MB per batch\n")

    # Start stats monitor (reads hailortcli monitor in background)
    stats = mon.StatsMonitor(interval=1.0)
    time.sleep(1.5)  # let first sample land

    dummy = np.zeros((input_h, input_w, 3), dtype=np.uint8)

    # Per-stage timing accumulators
    t_frame_read  = []
    t_tile_gen    = []
    t_full_pre    = []
    t_batch_stack = []
    t_infer       = []
    t_postproc    = []
    t_total       = []

    with VDevice() as device:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng  = device.configure(hef, cfg)[0]
        ivp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        ovp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

        with ng.activate():
            with InferVStreams(ng, ivp, ovp) as pipeline:

                total_frames = args.warmup + args.frames
                frame_idx    = 0
                recorded     = 0

                print(f"Running {args.warmup} warmup + {args.frames} measured frames...")

                while recorded < args.frames:
                    t0 = perf_counter()

                    # ── Frame read ──────────────────────────────────────────
                    t_a = perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        if hasattr(args, 'source') and str(args.source).endswith('.mp4'):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        if not ret:
                            break
                    t_b = perf_counter()

                    # ── Tile generation ──────────────────────────────────────
                    t_c = perf_counter()
                    tiles = make_tiles(frame, input_h, TILE_OVERLAP)
                    t_d = perf_counter()

                    # ── Full-frame preprocess ────────────────────────────────
                    t_e = perf_counter()
                    full_input, full_scale = preprocess(frame, input_h, input_w)
                    t_f = perf_counter()

                    # ── Batch stack + pad ────────────────────────────────────
                    t_g = perf_counter()
                    batch_imgs = [t[0] for t in tiles] + [full_input]
                    batch_imgs += [dummy] * (BATCH_SIZE - len(batch_imgs))
                    batch = np.stack(batch_imgs)
                    t_h = perf_counter()

                    # ── Hailo infer() ────────────────────────────────────────
                    # This is the single synchronous PCIe write → compute → PCIe read
                    t_i = perf_counter()
                    outputs = pipeline.infer({input_name: batch})
                    t_j = perf_counter()

                    out_list = outputs[output_name]

                    # ── Postprocess + NMS ────────────────────────────────────
                    t_k = perf_counter()
                    tile_scale = (1.0, 0, 0, input_h, input_w)
                    all_dets = []
                    for i, (_, tx, ty) in enumerate(tiles):
                        for d in postprocess({output_name: [out_list[i]]}, tile_scale, CONF):
                            d[0] += tx; d[2] += tx
                            d[1] += ty; d[3] += ty
                            all_dets.append(d)
                    full_out = {output_name: [out_list[n_tiles]]}
                    all_dets.extend(postprocess(full_out, full_scale, CONF))
                    nms(all_dets)
                    t_l = perf_counter()

                    t1 = perf_counter()
                    frame_idx += 1

                    if frame_idx <= args.warmup:
                        continue  # discard warmup frames

                    # Record timings (convert to ms)
                    t_frame_read .append((t_b - t_a) * 1000)
                    t_tile_gen   .append((t_d - t_c) * 1000)
                    t_full_pre   .append((t_f - t_e) * 1000)
                    t_batch_stack.append((t_h - t_g) * 1000)
                    t_infer      .append((t_j - t_i) * 1000)
                    t_postproc   .append((t_l - t_k) * 1000)
                    t_total      .append((t1 - t0)   * 1000)
                    recorded += 1

                    if recorded % 20 == 0:
                        print(f"  {recorded}/{args.frames} frames  "
                              f"infer={t_infer[-1]:.1f}ms  total={t_total[-1]:.1f}ms")

    cap.release()

    # ── Final stats from hailortcli monitor ─────────────────────────────────
    time.sleep(1.0)
    hw = stats.get()

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PIPELINE LATENCY BREAKDOWN  ({recorded} frames, "
          f"source {src_w}×{src_h})")
    print(f"{'='*70}")
    print(f"  {'Stage':<16}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  unit")
    print(f"  {'-'*60}")
    report("frame_read",    t_frame_read)
    report("tile_gen",      t_tile_gen)
    report("full_preproc",  t_full_pre)
    report("batch_stack",   t_batch_stack)

    cpu_pre = np.array(t_tile_gen) + np.array(t_full_pre) + np.array(t_batch_stack)
    report("CPU pre-infer", cpu_pre)

    print(f"  {'-'*60}")
    report("infer() total", t_infer)
    print(f"  {'-'*60}")
    report("postprocess",   t_postproc)
    report("FRAME TOTAL",   t_total)

    infer_arr  = np.array(t_infer)
    total_arr  = np.array(t_total)

    # Derived throughput
    mean_infer_s = infer_arr.mean() / 1000.0
    mean_total_s = total_arr.mean() / 1000.0
    tile_per_sec = n_infer / mean_infer_s
    frame_per_sec = 1.0 / mean_total_s

    # PCIe / Hailo compute estimate
    # Hailo reports FPS = batches/sec (each batch = BATCH_SIZE inferences)
    # On-chip compute per batch ≈ 1000 / hailo_fps ms (if hailo_fps available)
    hailo_fps = hw.get("hailo_fps")
    hailo_util = hw.get("hailo_device_util")
    cpu_pct    = hw.get("cpu_percent")

    print(f"\n{'='*70}")
    print(f"  THROUGHPUT SUMMARY")
    print(f"{'='*70}")
    print(f"  Inferences per batch      : {n_infer}  (tiled) + {BATCH_SIZE-n_infer} (pad)")
    print(f"  Effective tile/s          : {tile_per_sec:.1f}")
    print(f"  Effective frame/s         : {frame_per_sec:.1f}")
    print(f"  infer() mean              : {infer_arr.mean():.1f} ms")
    print(f"  CPU pre-infer mean        : {cpu_pre.mean():.1f} ms")
    print(f"  postprocess mean          : {np.array(t_postproc).mean():.1f} ms")

    print(f"\n{'='*70}")
    print(f"  HAILO HARDWARE METRICS (hailortcli monitor)")
    print(f"{'='*70}")
    if hailo_fps:
        on_chip_ms = 1000.0 / hailo_fps
        pcie_overhead_ms = infer_arr.mean() - on_chip_ms
        input_mb = input_bytes / 1024 / 1024
        # RPi5 PCIe Gen 2 x1 ≈ 500 MB/s practical
        pcie_bw_estimate = input_mb / max(pcie_overhead_ms / 1000.0, 1e-6) if pcie_overhead_ms > 0 else 0
        print(f"  Hailo FPS (hailortcli)    : {hailo_fps:.1f} batches/s")
        print(f"  On-chip compute estimate  : {on_chip_ms:.1f} ms  "
              f"(= 1000 / {hailo_fps:.1f})")
        print(f"  PCIe round-trip estimate  : {pcie_overhead_ms:.1f} ms  "
              f"(= infer - on-chip)")
        print(f"  Input data/batch          : {input_mb:.1f} MB")
        print(f"  Implied PCIe bandwidth    : {pcie_bw_estimate:.0f} MB/s")
    else:
        print(f"  Hailo FPS                 : n/a (hailortcli not available)")
        print(f"  Note: HAILO_MONITOR=1 env must be set before VDevice opens")
    print(f"  Hailo device utilization  : {hailo_util if hailo_util is not None else 'n/a'}%")
    print(f"  CPU utilization           : {cpu_pct if cpu_pct is not None else 'n/a'}%")

    print(f"\n{'='*70}")
    print(f"  DATA MARSHALING BREAKDOWN")
    print(f"{'='*70}")
    print(f"  batch_stack (np.stack)    : {np.array(t_batch_stack).mean():.2f} ms  "
          f"({input_bytes/1024/1024:.1f} MB  →  {input_bytes/1024/1024 / (np.array(t_batch_stack).mean()/1000):.0f} MB/s)")
    print(f"  Tile crop + pad           : {np.array(t_tile_gen).mean():.2f} ms")
    print(f"  Letterbox preprocess      : {np.array(t_full_pre).mean():.2f} ms")
    print()


if __name__ == "__main__":
    main()
