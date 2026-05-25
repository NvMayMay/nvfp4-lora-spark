#!/usr/bin/env python3
"""Pull per-step metrics from shipped training logs into a single JSON for plotting.

Why parse the log instead of reading training_progress.json? The on-disk JSON only carries
the loss tail (last 500); we need the full series (with cuda_alloc and elapsed wall time)
across all steps. The training script prints every 10 steps with a regex-friendly format,
so we recover the full picture from the log.

Outputs: train_metrics.json next to this script. Re-run any time the underlying logs grow.

Use `--root /path/to/research` for the directory containing the training log files
and `--adapters /path/to/adapters` for the directory containing adapter outputs.
"""
import argparse
import json, os, re, sys
from pathlib import Path


def build_runs(root: Path, adapters: Path) -> dict:
    return {
        "super_nvfp4": {
            "model": "Nemotron-3-Super-120B-A12B-NVFP4",
            "base_dtype": "nvfp4",
            "log": root / "train_super_nvfp4.log",
            "adapter": adapters / "nemotron_3_super_nvfp4_lora_ichv31_1epoch",
            "label": "Super-120B (NVFP4)",
            "color": "#1f77b4",
        },
        "nano_nvfp4": {
            "model": "Nemotron-3-Nano-30B-A3B-NVFP4",
            "base_dtype": "nvfp4",
            "log": root / "train_nano_nvfp4.log",
            "adapter": adapters / "nemotron_3_nano_nvfp4_lora_ichv31_1epoch",
            "label": "Nano-30B (NVFP4)",
            "color": "#2ca02c",
        },
        "nano_bf16": {
            "model": "Nemotron-3-Nano-30B-A3B-BF16",
            "base_dtype": "bf16",
            "log": root / "train_nano_bf16.log",
            "adapter": adapters / "nemotron_3_nano_bf16_lora_ichv31_1epoch",
            "label": "Nano-30B (BF16)",
            "color": "#d62728",
        },
    }

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+):\s+loss=([\d.]+)\s+avg20=([\d.]+)\s+elapsed=([\d.]+)m\s+eta=([\d.]+)h\s+cuda_alloc=([\d.]+)GB"
)
CKPT_RE = re.compile(r"\[checkpoint @ step (\d+)\]")
DONE_RE = re.compile(r"DONE:\s+(\d+)\s+steps,\s+wall=([\d.]+)h")


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    steps, losses, avg20s, elapsed_min, cuda_gb = [], [], [], [], []
    checkpoints = []
    total_steps = None
    wall_hours = None
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                s = int(m.group(1))
                total_steps = int(m.group(2))
                steps.append(s)
                losses.append(float(m.group(3)))
                avg20s.append(float(m.group(4)))
                elapsed_min.append(float(m.group(5)))
                cuda_gb.append(float(m.group(7)))
                continue
            m = CKPT_RE.search(line)
            if m:
                checkpoints.append(int(m.group(1)))
                continue
            m = DONE_RE.search(line)
            if m:
                wall_hours = float(m.group(2))

    if not steps:
        return {"status": "empty"}
    # per-step seconds (smoothed: differences in elapsed_min between consecutive step samples)
    sec_per_step = []
    for i in range(1, len(steps)):
        d_step = steps[i] - steps[i-1]
        d_min = elapsed_min[i] - elapsed_min[i-1]
        if d_step > 0:
            sec_per_step.append(d_min * 60 / d_step)
    median_step_s = sorted(sec_per_step)[len(sec_per_step)//2] if sec_per_step else None
    return {
        "status": "done" if wall_hours is not None else "in_progress",
        "total_steps_planned": total_steps,
        "last_step_seen": steps[-1],
        "wall_hours": wall_hours,
        "median_step_s": median_step_s,
        "final_avg20_loss": avg20s[-1],
        "min_avg20_loss": min(avg20s),
        "steps": steps,
        "losses": losses,
        "avg20": avg20s,
        "elapsed_min": elapsed_min,
        "cuda_alloc_gb": cuda_gb,
        "checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None,
                        help="directory containing train_super_nvfp4.log, train_nano_nvfp4.log, and train_nano_bf16.log")
    parser.add_argument("--adapters", type=Path, default=None,
                        help="directory containing adapter output subdirectories")
    args = parser.parse_args()
    if args.root is None or args.adapters is None:
        parser.error("--root and --adapters are required")

    out = {}
    for key, meta in build_runs(args.root, args.adapters).items():
        parsed = parse_log(meta["log"])
        out[key] = {**{k: v for k, v in meta.items() if k != "log" and k != "adapter"}, **parsed}
        out[key]["log"] = str(meta["log"])
        out[key]["adapter"] = str(meta["adapter"])

    dst = Path(__file__).parent / "train_metrics.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dst}")
    for key, run in out.items():
        if run.get("status") == "missing":
            print(f"  {key:14s} MISSING ({run.get('log','?')})")
        elif run.get("status") == "empty":
            print(f"  {key:14s} EMPTY (log exists but no step lines yet)")
        else:
            ckpts = len(run["checkpoints"])
            print(
                f"  {key:14s} {run['status']:11s} "
                f"step {run['last_step_seen']}/{run['total_steps_planned']}  "
                f"median {run['median_step_s']:.1f}s/step  "
                f"avg20 {run['final_avg20_loss']:.3f}  "
                f"ckpts={ckpts}"
                + (f"  wall={run['wall_hours']:.2f}h" if run["wall_hours"] else "")
            )


if __name__ == "__main__":
    main()
