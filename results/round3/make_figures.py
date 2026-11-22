#!/usr/bin/env python3
"""Figures of docs/REPORT.md section 5 (round 3) from the logs in this directory.

    python3 results/round3/make_figures.py      (needs matplotlib)
"""
import collections
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "docs", "figures")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def best(path, pattern, key_groups, val_group, take=min):
    d = collections.defaultdict(list)
    for line in open(path):
        m = re.match(pattern, line)
        if m:
            d[tuple(m.group(g) for g in key_groups)].append(float(m.group(val_group)))
    return {k: take(v) for k, v in d.items()}


def hbars(ax, labels, series, xlabel, title, ref=1.0, fmt="{:.2f}x"):
    n = len(series)
    h = 0.8 / n
    for s, (lab, col, vals) in enumerate(series):
        y = [i + (s - (n - 1) / 2) * h for i in range(len(labels))]
        ax.barh(y, [v - ref for v in vals], left=ref, height=h * 0.9, color=col, label=lab)
        for yy, v in zip(y, vals):
            ax.text(max(v, ref) + 0.01, yy, fmt.format(v), va="center", fontsize=7)
    ax.axvline(ref, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)


def fig_h100():
    path = os.path.join(HERE, "h100_omnicast.txt")
    if not os.path.exists(path):
        return
    b = best(path, r"(old|new) nb=(\d+) (\S+) \[(.*)\] ([\d.]+) us", (1, 2, 3, 4), 5)
    cases = [("fp32>e4m3", "--sat"), ("fp16>e4m3", "--sat"), ("bf16>e4m3", "--sat"), ("fp64>e4m3", "--sat"),
             ("fp32>e5m2", "--sat"), ("fp16>e5m2", "--sat"), ("fp32>e4m3", "--sat --scale 4"),
             ("fp16>e4m3", "--sat --trans"), ("fp32>e4m3", "--sat --trans"), ("fp64>e4m3", "--sat --trans"),
             ("fp64>e4m3", "--sat --dyn"), ("fp32>e4m3", "--sat --dyn"), ("fp64>e4m3", "--sat --trans --dyn"),
             ("fp32>e4m3", "--sat --trans --dyn"), ("fp16>e4m3", "--sat --trans --dyn"),
             ("fp32>e4m3", ""), ("fp64>fp32", ""), ("fp32>fp16", "")]
    def name(c, f):
        f = f.replace("--sat", "sat").replace("--scale 4", "scaled").replace("--trans", "transposed").replace("--dyn", "dynamic")
        return f"{c.replace('>', ' -> ')} {f}".strip()
    labels = [name(c, f) for c, f in cases]
    series = []
    for nb, col in (("2048", BLUE), ("8192", ORANGE)):
        series.append((f"{nb} x {nb}", col, [b[("old", nb, c, f)] / b[("new", nb, c, f)] for c, f in cases]))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    hbars(ax, labels, series, "speedup over 60a5cd1 (best of 20, 3 rounds)",
          "H100: FP8 converts of sm_89+ and the 4-column amax pass")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=2)
    fig.savefig(os.path.join(FIG, "round3_h100.png"))
    plt.close(fig)


def fig_dynamic():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    path = os.path.join(HERE, "h100_fused.txt")
    if os.path.exists(path):
        b = best(path, r"fused=(\d) nb=(\d+) (\S+) ([\d.]+) us", (1, 2, 3), 4)
        nbs = ["1024", "2048", "4096"]
        for src, col in (("fp64", BLUE), ("fp32", ORANGE), ("fp16", AQUA)):
            axs[0].plot(range(3), [b[("0", nb, src)] / b[("1", nb, src)] for nb in nbs], marker="o", markersize=6,
                        linewidth=2, color=col, label=f"{src} -> e4m3")
        axs[0].set_xticks(range(3))
        axs[0].set_xticklabels([f"{n}^2" for n in nbs])
        axs[0].axhline(1.0, color=INK2, linewidth=0.8)
        axs[0].set_ylabel("fused / two kernels (speedup)")
        axs[0].set_title("H100, transposed, dynamic scale", loc="left", fontsize=10)
        axs[0].legend(fontsize=8)
    r0 = best(os.path.join(HERE, "radeonvii_fused.txt"), r"fused=(\d) nb=(\d+) ([\d.]+) us", (1, 2), 3)
    r1 = best(os.path.join(HERE, "radeonvii_fused_amax_unrolled.txt"), r"fused=(\d) nb=(\d+) ([\d.]+) us", (1, 2), 3)
    nbs = ["1024", "2048", "4096", "8192"]
    x = range(len(nbs))
    for k, (lab, col, vals) in enumerate((
            ("amax pass, 4 columns per pass", BLUE, [r0[("0", n)] / r1[("0", n)] for n in nbs]),
            ("fused (OMNICAST_FUSED_DYNAMIC=1)", ORANGE, [r0[("0", n)] / r1[("1", n)] for n in nbs]))):
        axs[1].bar([i + (k - 0.5) * 0.38 for i in x], vals, width=0.36, color=col, label=lab)
        for i, v in zip(x, vals):
            axs[1].text(i + (k - 0.5) * 0.38, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    axs[1].axhline(1.0, color=INK2, linewidth=0.8)
    axs[1].set_xticks(list(x))
    axs[1].set_xticklabels([f"{n}^2" for n in nbs])
    axs[1].set_ylabel("speedup over 60a5cd1")
    axs[1].set_title("Radeon VII, fp64 -> e4m3", loc="left", fontsize=10)
    axs[1].set_ylim(0, 2.6)
    axs[1].legend(fontsize=7.5, loc="upper left")
    axs[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "round3_dynamic_scale.png"))
    plt.close(fig)


def fig_cpu_fp8():
    cpus = (("skylake", "Skylake-SP", BLUE), ("broadwell", "Broadwell", ORANGE), ("zen2", "Zen 2", AQUA))
    data = {c: best(os.path.join(HERE, f"cpu_fp8_encoders_{c}.txt"), r"(old|new) avx2 (\S+) (\S+) ([\d.]+) ns",
                    (1, 2, 3), 4) for c, _, _ in cpus}
    cases = [("fp16>e4m3", "nosat"), ("fp16>e4m3", "--sat"), ("fp32>e5m2", "nosat"), ("fp32>e4m3", "nosat"),
             ("fp64>e4m3", "nosat"), ("fp64>e5m2", "--sat"), ("fp64>e4m3", "--sat"), ("fp32>e5m2", "--sat"),
             ("fp32>e4m3", "--sat"), ("bf16>e4m3", "--sat")]
    labels = [f"{c.replace('>', ' -> ')} {'sat' if f == '--sat' else ''}".strip() for c, f in cases]
    series = [(name, col, [data[c][("old", k, f)] / data[c][("new", k, f)] for k, f in cases]) for c, name, col in cpus]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    hbars(ax, labels, series, "speedup over 60a5cd1 (ns per element, one thread, 262144 elements, AVX2 path)",
          "Host FP8 encoding: F16C rounding to odd, then the FP16 pattern")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG, "round3_cpu_fp8.png"))
    plt.close(fig)


def fig_cpu_inplace():
    cpus = (("broadwell", "Broadwell", ORANGE), ("zen2", "Zen 2", AQUA))
    data = {c: best(os.path.join(HERE, f"cpu_inplace_streaming_{c}.txt"), r"(old|new) inplace (\S+) ([\d.]+) GB/s",
                    (1, 2), 3, take=max) for c, _, _ in cpus}
    cases = ["fp32>fp64", "fp16>fp64", "fp16>fp32", "bf16>fp32"]
    series = [(name, col, [data[c][("new", k)] / data[c][("old", k)] for k in cases]) for c, name, col in cpus]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    hbars(ax, [c.replace(">", " -> ") for c in cases], series,
          "speedup over 60a5cd1 (in place, 2^25 elements, one thread)",
          "Host in place, growing types: streaming stores")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(FIG, "round3_cpu_inplace.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_h100()
    fig_dynamic()
    fig_cpu_fp8()
    fig_cpu_inplace()
    print("figures in", os.path.normpath(FIG))


if __name__ == "__main__":
    main()
