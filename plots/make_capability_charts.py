"""README capability charts. One coherent visual system.
Regenerate: python plots/make_capability_charts.py
Palette validated CVD-safe (dataviz skill); direct labels are the secondary encoding.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ---- design tokens (from dataviz reference palette, light surface) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LT = "#2a78d6", "#a9c9f1"
AQUA, YELLOW, ORANGE = "#1baf7a", "#eda100", "#eb6834"
RED, AMBER, GOODGREEN = "#d03b3b", "#fab219", "#006300"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "savefig.facecolor": SURFACE, "savefig.dpi": 150, "savefig.bbox": "tight",
})

def clean(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE); ax.spines["bottom"].set_color(BASELINE)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)

def titles(fig, t, sub, left=0.115, right=0.965, bottom=0.135, top=0.79):
    """Figure-level title + subtitle in reserved header space (no axes collision)."""
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.text(0.03, 0.955, t, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.03, 0.875, sub, fontsize=10.5, color=INK2, ha="left", va="top")

# =========================================================================
# 1. Spider lift across families (grouped bars: base vs adapter EM)
# =========================================================================
def chart1():
    models = ["Llama-3.1-8B", "Mistral-Small-24B", "Qwen3-32B"]
    base = [36.8, 24.5, 46.1]
    adapt = [52.0, 61.0, 43.4]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    x = range(len(models)); w = 0.36
    b1 = ax.bar([i - w/2 for i in x], base, w, color=BLUE_LT, label="4-bit base", zorder=3)
    b2 = ax.bar([i + w/2 for i in x], adapt, w, color=BLUE, label="+ NVFP4 LoRA", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}", (r.get_x()+r.get_width()/2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=11, color=INK, fontweight="bold")
    ax.annotate("EM saturated\n(NLL 1.29 -> 0.26)", (2, 46.1), xytext=(0, 26),
                textcoords="offset points", ha="center", fontsize=9.5, color=MUTED)
    clean(ax)
    ax.set_ylim(0, 74); ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylabel("Spider exact-set-match (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=2)
    titles(fig, "Same recipe lifts text-to-SQL across families",
           "Public Spider dev, trained and served on one GB10; adapter never merged")
    fig.savefig(f"{OUT}/spider_lift.png"); plt.close(fig)

# =========================================================================
# 2. Fits on one 128 GB box (training peak memory vs ceiling)
# =========================================================================
def chart2():
    labels = ["Nano-30B", "Super-120B", "Super-120B\n@ 262K ctx", "Super-120B\n@ 32K (frontier)"]
    mem = [36.1, 93.2, 90.8, 124.3]
    cols = [BLUE, BLUE, BLUE, AMBER]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    bars = ax.bar(range(len(labels)), mem, 0.6, color=cols, zorder=3)
    for r in bars:
        ax.annotate(f"{r.get_height():.0f} GB", (r.get_x()+r.get_width()/2, r.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=11, color=INK, fontweight="bold")
    ax.axhline(128, color=RED, linewidth=1.6, linestyle=(0, (5, 3)), zorder=4)
    ax.annotate("128 GB unified memory", (-0.45, 128), xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10.5, color=RED, fontweight="bold")
    clean(ax)
    ax.set_ylim(0, 150); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("Training peak memory (GB)")
    titles(fig, "A 120B model fine-tunes on one desk-side box",
           "NVFP4 LoRA training peak vs the 128 GB ceiling (amber = fit-tested frontier)")
    fig.savefig(f"{OUT}/fits_on_box.png"); plt.close(fig)

# =========================================================================
# 3. Any NVFP4 model (reach): horizontal bars by size, registered vs generic
# =========================================================================
def chart3():
    data = [
        ("Llama-3.1-8B", 8, "reg"), ("Mistral-Small-24B", 24, "reg"),
        ("Nemotron-Nano-30B", 30, "reg"), ("Qwen3-32B (dense)", 32, "reg"),
        ("Qwen3.6-35B", 35, "reg"), ("Command-A (cohere2)", 111, "gen"),
        ("Mistral-4-119B", 119, "reg"), ("Nemotron-Super-120B", 120, "reg"),
        ("Qwen3.5-122B", 122, "reg"),
    ]
    data.sort(key=lambda d: d[1])
    names = [d[0] for d in data]; sizes = [d[1] for d in data]
    cols = [ORANGE if d[2] == "gen" else BLUE for d in data]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y = range(len(names))
    bars = ax.barh(list(y), sizes, 0.62, color=cols, zorder=3)
    for r, s in zip(bars, sizes):
        ax.annotate(f"{s}B", (r.get_width(), r.get_y()+r.get_height()/2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=10.5, color=INK, fontweight="bold")
    clean(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=10.5, color=INK2)
    ax.set_xlim(0, 140); ax.set_xlabel("Parameters (billions)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="registered family (1 line each)"),
                       Patch(color=ORANGE, label="unregistered: generic fallback")],
              loc="lower right", frameon=False, fontsize=10.5)
    titles(fig, "Fine-tune any NVFP4 model, 8B to 122B",
           "Validated end to end on one GB10; Command-A has no registry entry", left=0.27)
    fig.savefig(f"{OUT}/reach_map.png"); plt.close(fig)

# =========================================================================
# 4. Batch, not single-stream (concurrency scaling, Nano-30B)
# =========================================================================
def chart4():
    conc = [1, 2, 4, 8]
    series = [
        ("512-tok prompt, 2048 out", [55.0, 116.1, 190.9, 338.9], BLUE),
        ("512 prompt, 256 out",      [54.5, 99.4, 205.5, 312.5], AQUA),
        ("512 prompt, 64 out",       [30.0, 90.8, 137.4, 199.8], YELLOW),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    for name, ys, c in series:
        ax.plot(conc, ys, "-o", color=c, linewidth=2.2, markersize=7, zorder=3, label=name)
        ax.annotate(f"{ys[-1]:.0f}", (conc[-1], ys[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=11,
                    color=INK, fontweight="bold")
    clean(ax)
    ax.set_xticks(conc); ax.set_xlim(0.6, 9.2); ax.set_ylim(0, 380)
    ax.set_xlabel("Concurrent requests"); ax.set_ylabel("Aggregate throughput (tok/s)")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    titles(fig, "The Spark is a batch box, not a single-stream box",
           "Nemotron-Nano-30B FT: 55 to 339 tok/s as concurrency scales 1 to 8")
    fig.savefig(f"{OUT}/concurrency.png"); plt.close(fig)

# =========================================================================
# 5. Command-A adapter applied at runtime (gold-SQL logprob before/after)
# =========================================================================
def chart5():
    labels = ["4-bit base", "+ NVFP4 LoRA\n(served, not merged)"]
    vals = [-28.8, -17.4]
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    bars = ax.bar(range(2), vals, 0.5, color=[BLUE_LT, BLUE], zorder=3)
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x()+r.get_width()/2, r.get_height()),
                    xytext=(0, -15), textcoords="offset points", ha="center",
                    fontsize=12, color=INK, fontweight="bold")
    ax.annotate("+11.4 nats", (0.5, -17.4), xytext=(0, 7), textcoords="offset points",
                ha="center", va="bottom", fontsize=13, color=GOODGREEN, fontweight="bold")
    ax.annotate("", xy=(0.5, -17.4), xytext=(0.5, -28.4),
                arrowprops=dict(arrowstyle="-|>", color=GOODGREEN, lw=1.8))
    clean(ax)
    ax.set_ylim(-34, 1); ax.axhline(0, color=BASELINE, linewidth=1.0)
    ax.set_xticks(range(2)); ax.set_xticklabels(labels)
    ax.set_ylabel("Gold-SQL log-prob (higher better)")
    titles(fig, "The fine-tune is actually applied at serve time",
           "Command-A (111B, unregistered): gold-SQL likelihood, base vs served adapter", left=0.10)
    fig.savefig(f"{OUT}/command_a_applied.png"); plt.close(fig)

for fn in (chart1, chart2, chart3, chart4, chart5):
    fn(); print(f"built {fn.__name__}")
print("OUT:", OUT)
