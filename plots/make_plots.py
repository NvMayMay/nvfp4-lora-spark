#!/usr/bin/env python3
"""Render training and serving plots for the model card.

Run modes:
  python make_plots.py all         # render every plot that has data
  python make_plots.py <plot_name> # render one (see PLOTS dict)
  python make_plots.py list        # show available plot names + data status

Training-side plots read `train_metrics.json` (produced by extract_train_metrics.py).
Serving-side plots read JSONL benchmark files under results/throughput_v1/ and
serve/diagnostics/.

Outputs land in plots/ as PNGs at dpi=150, suitable for README and HF model card embeds.
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
TRAIN_JSON = HERE / "train_metrics.json"

BENCH_FT_CUTLASS = REPO_ROOT / "results" / "throughput_v1" / "bench_merged_ft_20260524_110605.jsonl"
BENCH_BASE_CUTLASS = REPO_ROOT / "serve" / "diagnostics" / "bench_cutlass_eager_super_base_20260524_085637.jsonl"
BENCH_BASE_EMULATION = REPO_ROOT / "serve" / "diagnostics" / "bench_base_eager_emul_noblock_20260524_020643.jsonl"
SWEEP_NANO_CONCURRENCY = REPO_ROOT / "results" / "inference_concurrency_sweep" / "nano_ft_concurrency.json"
SWEEP_SUPER_NS3 = REPO_ROOT / "results" / "super_inference_concurrency_sweep" / "super_ft_ns3.json"

# 120 GB leaves about 8 GB of headroom for OS, driver, and CUDA pinned overhead.
SPARK_MEMORY_USABLE_BUDGET_GB = 120.0


def load_train():
    if not TRAIN_JSON.exists():
        raise SystemExit(f"missing {TRAIN_JSON} - run extract_train_metrics.py first")
    with open(TRAIN_JSON) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_data(run, need_steps=10):
    return run.get("status") != "missing" and len(run.get("steps", [])) >= need_steps


def plot_loss_curves(train):
    """3-run overlay of smoothed training loss vs step."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, run in train.items():
        if not has_data(run):
            continue
        ax.plot(run["steps"], run["avg20"], color=run["color"], lw=1.8, label=run["label"])
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (trailing-20 avg)")
    ax.set_title("LoRA training loss on ICH v3.1: Nemotron-3 family on DGX Spark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, "01_loss_curves.png")


def plot_quant_ablation_training(train):
    """Side-by-side: loss curve + cuda_alloc, for Nano-NVFP4 vs Nano-BF16."""
    nv = train["nano_nvfp4"]
    bf = train["nano_bf16"]
    if not has_data(nv) or not has_data(bf):
        return _stub("02_quant_ablation_training.png",
                     "Quant ablation plot needs both nano_nvfp4 and nano_bf16 runs.")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ylabel, field, title in [
        (axL, "training loss (avg20)",   "avg20",        "Loss"),
        (axR, "cuda allocated (GB)",     "cuda_alloc_gb", "Peak GPU memory"),
    ]:
        ax.plot(nv["steps"], nv[field], color=nv["color"], lw=1.6, label=nv["label"])
        ax.plot(bf["steps"], bf[field], color=bf["color"], lw=1.6, label=bf["label"])
        ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3); ax.legend(loc="best", framealpha=0.95)
    axR.axhline(SPARK_MEMORY_USABLE_BUDGET_GB, color="grey", ls="--", lw=1)
    axR.text(0.02, SPARK_MEMORY_USABLE_BUDGET_GB - 4, "Spark 120 GB usable budget",
             transform=axR.get_yaxis_transform(), fontsize=9, color="grey")
    fig.suptitle("Nano-30B quantization ablation: NVFP4 vs BF16 LoRA training", y=1.02)
    fig.tight_layout()
    return _save(fig, "02_quant_ablation_training.png")


def plot_memory_timeline(train):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, run in train.items():
        if not has_data(run):
            continue
        ax.plot(run["steps"], run["cuda_alloc_gb"], color=run["color"], lw=1.4, label=run["label"])
        for ck in run["checkpoints"]:
            ax.axvline(ck, color=run["color"], ls=":", lw=0.5, alpha=0.4)
    ax.axhline(SPARK_MEMORY_USABLE_BUDGET_GB, color="black", ls="--", lw=1)
    ax.text(0.99, SPARK_MEMORY_USABLE_BUDGET_GB + 1, "120 GB Spark usable budget",
            ha="right", color="black", fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_xlabel("step"); ax.set_ylabel("cuda allocated (GB)")
    ax.set_title(
        "GPU memory during LoRA training on DGX Spark (max_len=1536 + grad checkpointing)\n"
        "samples of cuda_alloc at log time; true peak is higher per the training table in README"
    )
    ax.set_ylim(0, SPARK_MEMORY_USABLE_BUDGET_GB + 10)
    ax.grid(True, alpha=0.3); ax.legend(loc="center right", framealpha=0.95)
    fig.tight_layout()
    return _save(fig, "03_memory_timeline.png")


def plot_throughput(train):
    runs = [(k, r) for k, r in train.items() if has_data(r)]
    if not runs:
        return _stub("04_throughput_and_memory.png", "Throughput plot needs at least one run with step data.")
    labels = [r["label"] for _, r in runs]
    sec_per_step = [r["median_step_s"] for _, r in runs]
    samples_per_hr = [3600 / s if s else 0 for s in sec_per_step]
    peak_mem = [max(r["cuda_alloc_gb"]) for _, r in runs]
    colors = [r["color"] for _, r in runs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    bars1 = axL.bar(labels, samples_per_hr, color=colors)
    axL.set_ylabel("training samples / hour")
    axL.set_title("Training throughput on Spark (batch=1, grad_accum=4, max_len=1536)")
    axL.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars1, samples_per_hr):
        axL.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10)

    bars2 = axR.bar(labels, peak_mem, color=colors)
    axR.axhline(SPARK_MEMORY_USABLE_BUDGET_GB, color="grey", ls="--", lw=1)
    axR.text(0.99, SPARK_MEMORY_USABLE_BUDGET_GB - 4, "120 GB usable budget",
             ha="right", color="grey", fontsize=9, transform=axR.get_yaxis_transform())
    axR.set_ylabel("peak cuda allocated (GB)")
    axR.set_title(
        "Sampled GPU memory during training\n"
        "(understates true peak by up to ~14 GB for the Nano-NVFP4 row)"
    )
    axR.set_ylim(0, SPARK_MEMORY_USABLE_BUDGET_GB + 10)
    axR.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars2, peak_mem):
        axR.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return _save(fig, "04_throughput_and_memory.png")


def plot_inference_throughput(_train_unused):
    """Per-workload-cell tok/s for Super-120B: base CUTLASS, merged-FT CUTLASS, base EMULATION."""
    base_cutlass = load_jsonl(BENCH_BASE_CUTLASS)
    ft_cutlass = load_jsonl(BENCH_FT_CUTLASS)
    base_emul = load_jsonl(BENCH_BASE_EMULATION)
    if not (base_cutlass and ft_cutlass and base_emul):
        missing = [str(p) for p in (BENCH_BASE_CUTLASS, BENCH_FT_CUTLASS, BENCH_BASE_EMULATION) if not p.exists()]
        return _stub("05_inference_throughput.png",
                     f"Inference throughput plot needs all 3 bench files. Missing: {missing}")

    cells = [(r["prompt_tokens_actual"], r["completion_tokens_actual"]) for r in base_cutlass]
    labels = [f"{p}+{o}" for p, o in cells]

    def by_cell(rows, cells):
        out = []
        for p, o in cells:
            match = next((r["tok_per_s"] for r in rows
                          if r["prompt_tokens_actual"] == p and r["completion_tokens_actual"] == o), None)
            out.append(match if match is not None else 0)
        return out

    base_y = by_cell(base_cutlass, cells)
    ft_y = by_cell(ft_cutlass, cells)
    emul_y = by_cell(base_emul, cells)

    n = len(cells)
    x = list(range(n))
    w = 0.27

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    bars1 = axL.bar([i - w for i in x], base_y, width=w, label="Super base (CUTLASS)", color="#9ecae1")
    bars2 = axL.bar(x,                   ft_y,   width=w, label="Super merged-FT (CUTLASS)", color="#1f77b4")
    bars3 = axL.bar([i + w for i in x], emul_y, width=w, label="Super base (EMULATION fallback)", color="#d62728")
    for bars, ys in ((bars1, base_y), (bars2, ft_y), (bars3, emul_y)):
        for b, v in zip(bars, ys):
            axL.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels(labels)
    axL.set_xlabel("workload (prompt + output tokens)")
    axL.set_ylabel("throughput (tok/s)")
    axL.set_title("Inference throughput on DGX Spark: Super-120B-NVFP4")
    axL.grid(True, axis="y", alpha=0.3)
    axL.legend(loc="upper right", framealpha=0.95)

    base_mean = sum(base_y) / len(base_y) if base_y else 0
    ft_mean = sum(ft_y) / len(ft_y) if ft_y else 0
    emul_mean = sum(emul_y) / len(emul_y) if emul_y else 0
    speedup = (ft_mean / emul_mean) if emul_mean else 0
    summary_means = [base_mean, ft_mean, emul_mean]
    summary_labels = ["base\nCUTLASS", "merged-FT\nCUTLASS", "base\nEMULATION"]
    summary_colors = ["#9ecae1", "#1f77b4", "#d62728"]
    sbars = axR.bar(summary_labels, summary_means, color=summary_colors)
    for b, v in zip(sbars, summary_means):
        axR.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=10)
    axR.set_ylabel("mean tok/s across cells")
    axR.set_title(f"Mean throughput (CUTLASS = {speedup:.0f}x EMULATION)")
    axR.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, "05_inference_throughput.png")


def plot_inference_concurrency(_train_unused):
    """Two-panel small multiples: short-prompt scaling vs long-prompt prefill saturation."""
    if not SWEEP_NANO_CONCURRENCY.exists():
        return _stub("06_inference_concurrency.png",
                     f"Concurrency plot needs {SWEEP_NANO_CONCURRENCY}")
    with open(SWEEP_NANO_CONCURRENCY) as f:
        data = json.load(f)

    # Left panel: short-prompt scaling (prompt=512) at ctx=4096, all 3 output lengths.
    short_tier = next((k for k in data if "ctx=4096" in k), next(iter(data)))
    short_rows = [r for r in data[short_tier] if r["prompt_tokens"] == 512]

    # Right panel: long-prompt prefill saturation - one curve per (tier, prompt_length),
    # taking only the longest output (2048) since prefill cost dominates and output length
    # has little leverage at that regime.
    long_curves = []
    for tier, rows in data.items():
        long_prompt = max(r["prompt_tokens"] for r in rows)
        long_rows = sorted([r for r in rows
                            if r["prompt_tokens"] == long_prompt and r["output_tokens"] == 2048],
                           key=lambda r: r["concurrency"])
        if long_rows:
            long_curves.append((long_prompt, long_rows))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: short-prompt scaling
    output_colors = {64: "#2ca02c", 256: "#1f77b4", 2048: "#d62728"}
    for o_len in [64, 256, 2048]:
        cells = sorted([r for r in short_rows if r["output_tokens"] == o_len],
                       key=lambda r: r["concurrency"])
        if not cells:
            continue
        xs = [r["concurrency"] for r in cells]
        ys = [r["comp_tok_per_s"] for r in cells]
        axL.plot(xs, ys, marker="o", lw=2.0, color=output_colors[o_len],
                 label=f"output={o_len}", markersize=7)
        for x, y in zip(xs, ys):
            axL.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color=output_colors[o_len])
    axL.set_xlabel("concurrency (simultaneous requests)")
    axL.set_ylabel("aggregate throughput (comp tok/s)")
    axL.set_xscale("log", base=2)
    axL.set_xticks([1, 2, 4, 8]); axL.set_xticklabels(["1", "2", "4", "8"])
    axL.set_title("Short prompt, input=512")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="upper left", title="output tokens", framealpha=0.95)
    axL.set_ylim(bottom=0)

    # Right: long-prompt prefill saturation
    prompt_colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    for i, (p_len, cells) in enumerate(long_curves):
        xs = [r["concurrency"] for r in cells]
        ys = [r["comp_tok_per_s"] for r in cells]
        axR.plot(xs, ys, marker="o", lw=2.0, color=prompt_colors[i % len(prompt_colors)],
                 label=f"prompt={p_len}", markersize=7)
        for x, y in zip(xs, ys):
            axR.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color=prompt_colors[i % len(prompt_colors)])
    axR.set_xlabel("concurrency (simultaneous requests)")
    axR.set_ylabel("aggregate throughput (comp tok/s)")
    axR.set_xscale("log", base=2)
    axR.set_xticks([1, 2, 4, 8]); axR.set_xticklabels(["1", "2", "4", "8"])
    axR.set_title("Long prompt, output=2048")
    axR.grid(True, alpha=0.3)
    axR.legend(loc="upper left", title="prompt tokens", framealpha=0.95)
    axR.set_ylim(bottom=0)

    fig.suptitle("Nano-30B-FT inference on Spark: concurrency scaling",
                 y=1.01, fontsize=12)
    fig.tight_layout()
    return _save(fig, "06_inference_concurrency.png")


def plot_super_inference_concurrency(_train_unused):
    """Super-120B-merged-FT aggregate tok/s vs concurrency. NS=3 is the deepest tier that fits."""
    if not SWEEP_SUPER_NS3.exists():
        return _stub("07_super_inference_concurrency.png",
                     f"Super concurrency plot needs {SWEEP_SUPER_NS3}")
    with open(SWEEP_SUPER_NS3) as f:
        data = json.load(f)
    trials = data.get("trials", [])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: prompt=512 scaling (clean compute case)
    output_colors = {64: "#2ca02c", 256: "#1f77b4", 2048: "#d62728"}
    for o_len in [64, 256, 2048]:
        cells = sorted([r for r in trials if r["prompt_tokens"] == 512 and r["output_tokens"] == o_len],
                       key=lambda r: r["n_concurrent"])
        if not cells:
            continue
        xs = [r["n_concurrent"] for r in cells]
        ys = [r["agg_completion_tps"] for r in cells]
        axL.plot(xs, ys, marker="o", lw=2.0, color=output_colors[o_len],
                 label=f"output={o_len}", markersize=8)
        for x, y in zip(xs, ys):
            axL.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color=output_colors[o_len])
    axL.set_xlabel("concurrency (simultaneous requests)")
    axL.set_ylabel("aggregate throughput (comp tok/s)")
    axL.set_xticks([1, 2, 3])
    axL.set_title("Super-120B-FT, prompt=512 in: 2.9x peak speedup at conc=3")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="upper left", title="output tokens", framealpha=0.95)
    axL.set_ylim(bottom=0)

    # Right: prompt=2048 scaling (prefill-loaded case)
    for o_len in [64, 256, 2048]:
        cells = sorted([r for r in trials if r["prompt_tokens"] == 2048 and r["output_tokens"] == o_len],
                       key=lambda r: r["n_concurrent"])
        if not cells:
            continue
        xs = [r["n_concurrent"] for r in cells]
        ys = [r["agg_completion_tps"] for r in cells]
        axR.plot(xs, ys, marker="o", lw=2.0, color=output_colors[o_len],
                 label=f"output={o_len}", markersize=8)
        for x, y in zip(xs, ys):
            axR.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color=output_colors[o_len])
    axR.set_xlabel("concurrency (simultaneous requests)")
    axR.set_ylabel("aggregate throughput (comp tok/s)")
    axR.set_xticks([1, 2, 3])
    axR.set_title("Super-120B-FT, prompt=2048 in: 2.6x peak speedup at conc=3")
    axR.grid(True, alpha=0.3)
    axR.legend(loc="upper left", title="output tokens", framealpha=0.95)
    axR.set_ylim(bottom=0)

    fig.suptitle("Super-120B-NVFP4 merged-FT inference on Spark: concurrency scales 2-3x via CUTLASS",
                 y=1.01, fontsize=12)
    fig.tight_layout()
    return _save(fig, "07_super_inference_concurrency.png")


def _save(fig, name):
    path = HERE / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _stub(name, reason):
    """Placeholder image with a why-no-data message."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, fontsize=11)
    ax.set_axis_off()
    return _save(fig, name)


PLOTS = {
    "loss_curves":                  plot_loss_curves,
    "quant_ablation_training":      plot_quant_ablation_training,
    "memory_timeline":              plot_memory_timeline,
    "throughput":                   plot_throughput,
    "inference_throughput":         plot_inference_throughput,
    "inference_concurrency":        plot_inference_concurrency,
    "super_inference_concurrency":  plot_super_inference_concurrency,
}


def main(argv):
    if len(argv) < 2 or argv[1] == "list":
        print("available plots:")
        for name in PLOTS:
            print(f"  {name}")
        return 0

    train = load_train()
    targets = list(PLOTS.keys()) if argv[1] == "all" else [argv[1]]
    print(f"rendering {len(targets)} plot(s)...")
    for name in targets:
        if name not in PLOTS:
            print(f"  skip unknown: {name}")
            continue
        PLOTS[name](train)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
