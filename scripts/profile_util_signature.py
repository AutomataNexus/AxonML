#!/usr/bin/env python3
"""
util-signature sampler for AxonML training perf diagnosis (deficiency #1).

Captures high-frequency "utilization signature" (GPU util, VRAM, power, CPU threads,
per-process RSS) while a training process is running. The goal is to distinguish:

- Pure CPU graph-walk / autograd dispatch overhead (high single-thread CPU, low VRAM, low GPU util)
- VRAM oversubscription + WSL/WDDM host-spill (VRAM pegged near 12 GB ceiling, sudden 100x+ slowdown
  in identical MatMulBackward kernels once hot, possible host page faults visible as CPU sys time spikes)

Usage (two terminals):

  Terminal A (sampler):
    python3 scripts/profile_util_signature.py --interval 0.05 --duration 300 --out /tmp/util_sig_$(date +%s).csv

  Terminal B (training, with backward profiling):
    AXONML_PROFILE_BACKWARD=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib \
      cargo run --release --features cuda --example simple_training 2>&1 | tee /tmp/train.log

Or target a specific PID:
    python3 scripts/profile_util_signature.py --pid 12345 --interval 0.02 --out sig.csv

The sampler logs CSV with:
  ts, gpu_util, mem_util, vram_used_mb, vram_total_mb, power_w, sm_clock, cpu_user, cpu_sys, rss_mb, threads, note

After run, look for the moment the [backward profile] shows MatMulBackward jumping from ~1ms to ~300+ ms.
Correlate the timestamp with VRAM.used crossing ~11.5 GB and GPU.util dropping.

If VRAM peg + blowup coincide, and CPU is one thread at 100% R during the long ops (not the graph walk), this confirms host-spill / WDDM paging, not "CPU graph walk" (L82 hypothesis).

Requires: nvidia-smi (always), optional psutil for richer CPU/thread data.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime


def get_nvidia_stats():
    """Return dict of gpu stats from nvidia-smi or None on fail."""
    try:
        # -l 0 would loop in nvidia-smi; we drive the loop here for sync with other samplers.
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2)
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            return None
        return {
            "gpu_util": float(parts[0]),
            "mem_util": float(parts[1]),
            "vram_used_mb": float(parts[2]),
            "vram_total_mb": float(parts[3]),
            "power_w": float(parts[4]),
            "sm_clock": float(parts[5]),
        }
    except Exception:
        return None


def get_process_stats(pid):
    """Return CPU/RSS stats for pid (and its threads if possible). Best-effort."""
    stats = {"cpu_user": 0.0, "cpu_sys": 0.0, "rss_mb": 0.0, "threads": 0, "note": ""}
    try:
        # /proc/<pid>/stat for whole process (jiffies, convert rough)
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().strip().split()
            # utime = 14, stime=15 (0-based after pid etc), rss=23 (pages)
            if len(fields) > 23:
                utime = int(fields[13])
                stime = int(fields[14])
                rss_pages = int(fields[23])
                stats["cpu_user"] = utime / 100.0  # rough, ticks -> sec assuming 100hz
                stats["cpu_sys"] = stime / 100.0
                stats["rss_mb"] = (rss_pages * 4) / 1024.0  # 4k pages typical
        # threads count
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    stats["threads"] = int(line.split()[1])
                    break
        # Try psutil for better per-thread if available (non-fatal)
        try:
            import psutil  # type: ignore

            p = psutil.Process(pid)
            cpu = p.cpu_times()
            stats["cpu_user"] = cpu.user
            stats["cpu_sys"] = cpu.system
            stats["rss_mb"] = p.memory_info().rss / (1024 * 1024)
            stats["threads"] = p.num_threads()
            # If many threads, note the top consumer (heuristic for "one thread at 100%")
            if stats["threads"] > 4:
                stats["note"] = "multi-thread"
        except Exception:
            stats["note"] = stats.get("note", "") + ";no-psutil"
    except FileNotFoundError:
        stats["note"] = "pid-not-found"
    except Exception as e:
        stats["note"] = f"err:{type(e).__name__}"
    return stats


def find_training_pid():
    """Heuristic: find a likely AxonML/CUDA training process."""
    try:
        out = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
        candidates = []
        for line in out.splitlines():
            if "cargo" in line or "train" in line or "example" in line or "llm-training" in line:
                if "cuda" in line.lower() or "--features" in line or "release" in line:
                    parts = line.split()
                    if len(parts) > 1:
                        candidates.append(int(parts[1]))
        if candidates:
            return candidates[0]
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="AxonML util-signature sampler for training perf blowup diagnosis")
    ap.add_argument("--pid", type=int, default=None, help="PID of training process (auto-detect if omitted)")
    ap.add_argument("--interval", type=float, default=0.05, help="Sample interval seconds (0.02-0.1 typical)")
    ap.add_argument("--duration", type=float, default=300, help="Max seconds to sample")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default /tmp/util_sig_YYYYMMDD_HHMMSS.csv)")
    args = ap.parse_args()

    pid = args.pid
    if pid is None:
        pid = find_training_pid()
        if pid:
            print(f"[sampler] auto-detected training PID {pid}")
        else:
            print("[sampler] no --pid and no auto-detect; will still sample GPU globally (process stats will be 0)")
            pid = 0

    if args.out:
        out_path = args.out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"/tmp/util_sig_{ts}.csv"

    print(f"[sampler] logging to {out_path}  interval={args.interval}s  duration={args.duration}s")
    print("[sampler] Columns: ts_iso,gpu_util,mem_util,vram_used_mb,vram_total_mb,power_w,sm_clock,cpu_user,cpu_sys,rss_mb,threads,note")
    print("[sampler] Run training in parallel with AXONML_PROFILE_BACKWARD=1 + LD_LIBRARY_PATH=/usr/lib/wsl/lib")
    print("[sampler] Watch for MatMulBackward time jump in the other terminal's stderr; note the wall time and compare to this log.")

    fieldnames = [
        "ts_iso", "gpu_util", "mem_util", "vram_used_mb", "vram_total_mb",
        "power_w", "sm_clock", "cpu_user", "cpu_sys", "rss_mb", "threads", "note"
    ]

    start = time.time()
    samples = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        while True:
            now = time.time()
            if now - start > args.duration:
                break

            gpu = get_nvidia_stats() or {
                "gpu_util": -1, "mem_util": -1, "vram_used_mb": -1, "vram_total_mb": -1,
                "power_w": -1, "sm_clock": -1
            }
            proc = get_process_stats(pid) if pid else {"cpu_user": 0, "cpu_sys": 0, "rss_mb": 0, "threads": 0, "note": "no-pid"}

            row = {
                "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
                **gpu,
                **proc,
            }
            w.writerow(row)
            f.flush()
            samples += 1

            # Sleep precisely
            time.sleep(max(0.001, args.interval - (time.time() - now)))

    print(f"[sampler] done. {samples} samples -> {out_path}")
    print("[sampler] Post-process tip: awk -F, 'NR>1 { if ($4 > 11500) print $1, \"VRAM>\", $4 }' the csv")
    print("[sampler] or python -c 'import pandas as pd; df=pd.read_csv(...) ; print(df[df.vram_used_mb>11000].head())'")


if __name__ == "__main__":
    main()
