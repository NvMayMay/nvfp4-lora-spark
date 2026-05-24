#!/usr/bin/env python3
"""Render all candidate plots for the GH/HF model card.

Run modes:
  python make_plots.py all         # render every plot that has data
  python make_plots.py <plot_name> # render one (see PLOTS dict)
  python make_plots.py list        # show available plot names + data status

Training-side plots read `train_metrics.json` (produced by extract_train_metrics.py).
Eval-side plots read `eval_results.json` (format documented at bottom of this file).

Outputs land in plots/ as PNGs. PNG dpi=150 - fine for README embeds and HF cards.
"""
import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TRAIN_JSON = HERE / "train_metrics.json"
EVAL_JSON = HERE / "eval_results.json"

# Spark hardware constants for annotations
SPARK_MEMORY_CEILING_GB = 120.0

# ── helpers ────────────────────────────────────────────────────────────────────

def load_train():
    if not TRAIN_JSON.exists():
        raise SystemExit(f"missing {TRAIN_JSON} - run extract_train_metrics.py first")
    with open(TRAIN_JSON) as f:
        return json.load(f)


def load_eval():
    if not EVAL_JSON.exists():
        return None
    with open(EVAL_JSON) as f:
        return json.load(f)


def smooth(xs, window=20):
    """Trailing window mean - matches the avg20 the training script logged."""
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i+1]) / (i - lo + 1))
    return out


def has_data(run, need_steps=10):
    return run.get("status") != "missing" and len(run.get("steps", [])) >= need_steps


# ── plot 1: loss curves overlay ────────────────────────────────────────────────

def plot_loss_curves(train):
    """3-run overlay of smoothed training loss vs step."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, run in train.items():
        if not has_data(run): continue
        ax.plot(run["steps"], run["avg20"], color=run["color"], lw=1.8, label=run["label"])
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (trailing-20 avg)")
    ax.set_title("LoRA training loss on ICH v3.1 - Nemotron-3 family on DGX Spark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, "01_loss_curves.png")


# ── plot 2: quant-ablation (training side, Nano NVFP4 vs BF16) ─────────────────

def plot_quant_ablation_training(train):
    """Side-by-side: loss curve + cuda_alloc, for Nano-NVFP4 vs Nano-BF16."""
    nv = train["nano_nvfp4"]; bf = train["nano_bf16"]
    if not has_data(nv) or not has_data(bf):
        return _stub("02_quant_ablation_training.png",
                     "Quant ablation plot needs both nano_nvfp4 and nano_bf16 runs (Day 5b + Day 5c).")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ylabel, field, title in [
        (axL, "training loss (avg20)",   "avg20",        "Loss"),
        (axR, "cuda allocated (GB)",     "cuda_alloc_gb", "Peak GPU memory"),
    ]:
        ax.plot(nv["steps"], nv[field], color=nv["color"], lw=1.6, label=nv["label"])
        ax.plot(bf["steps"], bf[field], color=bf["color"], lw=1.6, label=bf["label"])
        ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3); ax.legend(loc="best", framealpha=0.95)
    axR.axhline(SPARK_MEMORY_CEILING_GB, color="grey", ls="--", lw=1)
    axR.text(0.02, SPARK_MEMORY_CEILING_GB - 4, "Spark 120 GB ceiling",
             transform=axR.get_yaxis_transform(), fontsize=9, color="grey")
    fig.suptitle("Nano-30B quantization ablation: NVFP4 vs BF16 LoRA training", y=1.02)
    fig.tight_layout()
    return _save(fig, "02_quant_ablation_training.png")


# ── plot 3: memory timeline with checkpoints + ceiling ─────────────────────────

def plot_memory_timeline(train):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, run in train.items():
        if not has_data(run): continue
        ax.plot(run["steps"], run["cuda_alloc_gb"], color=run["color"], lw=1.4, label=run["label"])
        for ck in run["checkpoints"]:
            ax.axvline(ck, color=run["color"], ls=":", lw=0.5, alpha=0.4)
    ax.axhline(SPARK_MEMORY_CEILING_GB, color="black", ls="--", lw=1)
    ax.text(0.99, SPARK_MEMORY_CEILING_GB + 1, "120 GB Spark ceiling",
            ha="right", color="black", fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_xlabel("step"); ax.set_ylabel("cuda allocated (GB)")
    ax.set_title("GPU memory during LoRA training on DGX Spark (max_len=1536 + grad checkpointing)")
    ax.set_ylim(0, SPARK_MEMORY_CEILING_GB + 10)
    ax.grid(True, alpha=0.3); ax.legend(loc="center right", framealpha=0.95)
    fig.tight_layout()
    return _save(fig, "03_memory_timeline.png")


# ── plot 4: throughput + peak memory bar chart ────────────────────────────────

def plot_throughput(train):
    runs = [(k, r) for k, r in train.items() if has_data(r)]
    if not runs:
        return _stub("04_throughput_and_memory.png", "Throughput plot needs ≥1 run with step data.")
    labels = [r["label"] for _, r in runs]
    sec_per_step = [r["median_step_s"] for _, r in runs]
    samples_per_hr = [3600 / s if s else 0 for s in sec_per_step]
    peak_mem = [max(r["cuda_alloc_gb"]) for _, r in runs]
    colors = [r["color"] for _, r in runs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    bars1 = axL.bar(labels, samples_per_hr, color=colors)
    axL.set_ylabel("training samples / hour")
    axL.set_title("Throughput on Spark (batch=1, grad_accum=4, max_len=1536)")
    axL.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars1, samples_per_hr):
        axL.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10)

    bars2 = axR.bar(labels, peak_mem, color=colors)
    axR.axhline(SPARK_MEMORY_CEILING_GB, color="grey", ls="--", lw=1)
    axR.text(0.99, SPARK_MEMORY_CEILING_GB - 4, "120 GB ceiling",
             ha="right", color="grey", fontsize=9, transform=axR.get_yaxis_transform())
    axR.set_ylabel("peak cuda allocated (GB)")
    axR.set_title("Peak GPU memory during training")
    axR.set_ylim(0, SPARK_MEMORY_CEILING_GB + 10)
    axR.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars2, peak_mem):
        axR.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return _save(fig, "04_throughput_and_memory.png")


# ── plot 5: base-vs-FT accuracy (Day 6 eval data required) ─────────────────────

def plot_eval_headline(eval_data, train):
    """Grouped bars: one group per task, bars = {nano_base, nano_ft, super_base, super_ft}."""
    if eval_data is None:
        return _stub("05_eval_headline_accuracy.png",
                     "Headline accuracy needs eval_results.json (post-Day 6). See expected schema in make_plots.py.")
    tasks = list(eval_data.keys())
    model_keys = ["nano_base_nvfp4", "nano_ft_nvfp4", "super_base_nvfp4", "super_ft_nvfp4"]
    model_labels = ["Nano base", "Nano +FT", "Super base", "Super +FT"]
    colors = ["#7fcdbb", "#2ca02c", "#9ecae1", "#1f77b4"]

    n_tasks = len(tasks); n_models = len(model_keys)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_tasks + 4), 5.5))
    for i, (mk, ml, col) in enumerate(zip(model_keys, model_labels, colors)):
        vals = [eval_data[t].get(mk) for t in tasks]
        xs = [j + (i - n_models/2 + 0.5) * width for j in range(n_tasks)]
        valid_xs = [x for x, v in zip(xs, vals) if v is not None]
        valid_vs = [v for v in vals if v is not None]
        ax.bar(valid_xs, valid_vs, width=width, label=ml, color=col)
    ax.set_xticks(range(n_tasks)); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Eval accuracy: base vs LoRA-FT, Nano vs Super (NVFP4 + vLLM marlin on Spark)")
    ax.legend(loc="best", framealpha=0.95); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "05_eval_headline_accuracy.png")


# ── plot 6: FT lift by model size ──────────────────────────────────────────────

def plot_eval_ft_lift(eval_data, train):
    if eval_data is None:
        return _stub("06_eval_ft_lift.png", "FT-lift plot needs eval_results.json (post-Day 6).")
    tasks = list(eval_data.keys())
    nano_lift = []; super_lift = []
    for t in tasks:
        d = eval_data[t]
        nano_lift.append((d.get("nano_ft_nvfp4") or 0) - (d.get("nano_base_nvfp4") or 0))
        super_lift.append((d.get("super_ft_nvfp4") or 0) - (d.get("super_base_nvfp4") or 0))
    x = range(len(tasks)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tasks) + 4), 5.5))
    ax.bar([i - w/2 for i in x], nano_lift, width=w, color="#2ca02c", label="Nano-30B lift")
    ax.bar([i + w/2 for i in x], super_lift, width=w, color="#1f77b4", label="Super-120B lift")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("Δ accuracy  (FT − base)")
    ax.set_title("LoRA fine-tuning lift on ICH v3.1, by model size")
    ax.legend(loc="best", framealpha=0.95); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "06_eval_ft_lift.png")


# ── plot 7: quant tax on eval (Nano NVFP4 vs Nano BF16) ─────────────────────────

def plot_eval_quant_tax(eval_data, train):
    if eval_data is None:
        return _stub("07_eval_quant_tax.png", "Quant-tax-on-eval plot needs eval_results.json (post-Day 6 with Day 5c row).")
    tasks = list(eval_data.keys())
    nvfp4 = [eval_data[t].get("nano_ft_nvfp4") for t in tasks]
    bf16  = [eval_data[t].get("nano_ft_bf16")  for t in tasks]
    if all(v is None for v in bf16):
        return _stub("07_eval_quant_tax.png", "Need Day 5c BF16 FT in eval_results.json before quant-tax plot can render.")
    x = range(len(tasks)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tasks) + 4), 5.5))
    ax.bar([i - w/2 for i in x], [v or 0 for v in nvfp4], width=w, color="#2ca02c", label="Nano-FT (NVFP4 base)")
    ax.bar([i + w/2 for i in x], [v or 0 for v in bf16],  width=w, color="#d62728", label="Nano-FT (BF16 base)")
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Quantization tax on adapter quality: Nano LoRA-FT on NVFP4 base vs BF16 base")
    ax.legend(loc="best", framealpha=0.95); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "07_eval_quant_tax.png")


# ── boilerplate ────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = HERE / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _stub(name, reason):
    """Render a placeholder image saying why the plot can't be made yet."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, fontsize=11)
    ax.set_axis_off()
    return _save(fig, name)


PLOTS = {
    "loss_curves":              ("training", plot_loss_curves),
    "quant_ablation_training":  ("training", plot_quant_ablation_training),
    "memory_timeline":          ("training", plot_memory_timeline),
    "throughput":               ("training", plot_throughput),
    "eval_headline":            ("eval",     plot_eval_headline),
    "eval_ft_lift":             ("eval",     plot_eval_ft_lift),
    "eval_quant_tax":           ("eval",     plot_eval_quant_tax),
}


def main(argv):
    if len(argv) < 2 or argv[1] == "list":
        print("available plots:")
        for name, (kind, _) in PLOTS.items():
            print(f"  {name:30s} ({kind})")
        return 0

    train = load_train()
    eval_data = load_eval()
    targets = list(PLOTS.keys()) if argv[1] == "all" else [argv[1]]
    print(f"rendering {len(targets)} plot(s)...")
    for name in targets:
        if name not in PLOTS: print(f"  skip unknown: {name}"); continue
        kind, fn = PLOTS[name]
        if kind == "eval":
            fn(eval_data, train)
        else:
            fn(train)
    return 0


# ── expected eval_results.json schema ──────────────────────────────────────────
# {
#   "<task_name_1>": {
#     "nano_base_nvfp4": 0.45,
#     "nano_ft_nvfp4":   0.62,
#     "super_base_nvfp4": 0.51,
#     "super_ft_nvfp4":   0.68,
#     "nano_ft_bf16":     0.63    # optional, only required for plot 07
#   },
#   "<task_name_2>": {...},
#   ...
# }
# Tasks = whatever subdivisions your eval harness exports (e.g. exact_match, semantic,
# per-subscore). Missing keys are skipped per-bar, not per-plot.


if __name__ == "__main__":
    sys.exit(main(sys.argv))
