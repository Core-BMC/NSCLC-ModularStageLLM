"""Main Figure 2 (A-plan): triangle radars + Delta bars, faithful to the original
layout but rebuilt on the standardized re-run (3 models, AJCC 8th & 9th, corrected
baseline). Radars use accuracy; Delta bars show F1 and accuracy improvement (MAA - baseline).
Reads results/table4.csv, table4_f1.csv. Run: python3 make_figure2_main.py
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")

MODELS = ["GPT-4o", "LLaMA 3.3 70B", "Phi-4 14B"]
SHORT = ["GPT-4o", "LLaMA 3.3 70B", "Phi-4 14B"]
EDS = ["8th", "9th"]
AXES = ["T", "N", "M"]

COLOR_BASE = "#ff6b6b"   # baseline (red, dashed)
COLOR_MAA = "#51cf66"    # MAA (green, solid)
MODEL_COLORS = ["#4dabf7", "#ff922b", "#cc5de8"]  # GPT-4o, LLaMA, Phi-4
RMIN = 0.40              # radial axis floor (accuracy); disclosed in caption


def load(fn):
    with open(os.path.join(RES, fn)) as f:
        return list(csv.DictReader(f))


ACC = {(r["model"], r["mode"], r["edition"], r["axis"]): float(r["accuracy"]) / 100
       for r in load("table4.csv")}
F1 = {(r["model"], r["mode"], r["edition"], r["axis"]): float(r["macro_f1"]) / 100
      for r in load("table4_f1.csv")}
# paired baseline-vs-MAA deltas with bootstrap 95% CI and significance p (from compute_sig.py)
SIG = {(r["model"], r["edition"], r["axis"]): {k: float(r[k]) for k in
       ("acc_delta", "acc_dlo", "acc_dhi", "acc_p", "f1_delta", "f1_dlo", "f1_dhi", "f1_p")}
       for r in load("sig.csv")}


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def radar(ax, base, maa, title):
    ang = np.array([90, 330, 210]) * np.pi / 180  # N top, T right, M left

    def rr(v):  # map accuracy (fraction) to plotted radius, floor = RMIN
        return max(0.0, (v - RMIN) / (1.0 - RMIN))

    for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:  # gridlines labeled with true accuracy (%)
        ax.add_patch(plt.Circle((0, 0), rr(v), color="gray", fill=False, lw=0.5, alpha=0.3))
        ax.text(0.03, -rr(v), f"{int(v * 100)}", fontsize=6, color="gray", ha="left", va="center")
    ax.text(0, 1.15, "N", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(1.15 * np.cos(ang[1]), 1.15 * np.sin(ang[1]), "T", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(1.15 * np.cos(ang[2]), 1.15 * np.sin(ang[2]), "M", ha="center", va="center", fontsize=11, weight="bold")
    for a in ang:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], "gray", lw=0.8, alpha=0.5)

    def coords(v):  # v = {T,N,M}
        return np.array([[rr(v["N"]) * np.cos(ang[0]), rr(v["N"]) * np.sin(ang[0])],
                         [rr(v["T"]) * np.cos(ang[1]), rr(v["T"]) * np.sin(ang[1])],
                         [rr(v["M"]) * np.cos(ang[2]), rr(v["M"]) * np.sin(ang[2])]])
    bc, mc = coords(base), coords(maa)
    ax.add_patch(Polygon(bc, fill=True, alpha=0.15, edgecolor=COLOR_BASE, facecolor=COLOR_BASE, lw=2.2, ls="--", zorder=2))
    ax.add_patch(Polygon(mc, fill=True, alpha=0.15, edgecolor=COLOR_MAA, facecolor=COLOR_MAA, lw=2.8, ls="-", zorder=3))
    for c, col, ls, mk in [(bc, COLOR_BASE, "--", "o"), (mc, COLOR_MAA, "-", "s")]:
        outline = np.vstack([c, c[0]])
        ax.plot(outline[:, 0], outline[:, 1], color=col, lw=2.3, ls=ls, zorder=4)
        ax.scatter(c[:, 0], c[:, 1], s=55, facecolor="white", edgecolor=col, lw=1.4, zorder=6, marker=mk)
    ax.text(0, -1.32, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.45, 1.3); ax.set_aspect("equal"); ax.axis("off")


def delta_panel(ax, metric, ed, title, letter):
    """metric = 'acc' or 'f1'. Bars = MAA - baseline; whiskers = bootstrap 95% CI;
    stars = paired significance (McNemar for accuracy, bootstrap for F1)."""
    x = np.arange(3); w = 0.26
    extremes = [0.0]
    for i, (model, col, name) in enumerate(zip(MODELS, MODEL_COLORS, SHORT)):
        d, lo, hi, ps = [], [], [], []
        for a in AXES:
            s = SIG[(model, ed, a)]
            dv, dlo, dhi = s[metric + "_delta"], s[metric + "_dlo"], s[metric + "_dhi"]
            d.append(dv); lo.append(dv - dlo); hi.append(dhi - dv); ps.append(s[metric + "_p"])
        ax.bar(x + i * w, d, w, label=name, color=col, edgecolor="black", lw=0.7, alpha=0.9,
               yerr=[lo, hi], capsize=3, error_kw=dict(lw=1, ecolor="#333333"))
        for k in range(3):
            if d[k] >= 0:
                yt, va, off = d[k] + hi[k], "bottom", 0.004
            else:
                yt, va, off = d[k] - lo[k], "top", -0.004
            ax.text(x[k] + i * w, yt + off, star(ps[k]), ha="center", va=va, fontsize=7)
            extremes += [d[k] + hi[k], d[k] - lo[k]]
    ax.set_xlabel("Category", fontsize=11, weight="bold")
    ax.set_ylabel("Improvement (Δ)", fontsize=11, weight="bold")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xticks(x + w); ax.set_xticklabels(AXES, fontsize=10)
    ax.axhline(0, color="black", lw=1, alpha=0.35)
    ax.grid(axis="y", alpha=0.2, ls="--")
    lo_y, hi_y = min(extremes), max(extremes)
    m = (hi_y - lo_y) * 0.12
    ax.set_ylim(lo_y - m, hi_y + m * 1.6)
    ax.legend(fontsize=8.5, frameon=False, loc="best", ncol=3)
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=15, weight="bold", va="top")


def main():
    plt.rcParams["font.size"] = 11
    fig = plt.figure(figsize=(15, 17), facecolor="white")
    gs_top = fig.add_gridspec(2, 3, left=0.07, right=0.97, top=0.955, bottom=0.53, wspace=0.28, hspace=0.26)
    gs_bot = fig.add_gridspec(2, 2, left=0.09, right=0.97, top=0.45, bottom=0.05, wspace=0.26, hspace=0.34)

    # Panel A: radars (rows = editions, cols = models)
    for ri, ed in enumerate(EDS):
        for ci, (model, name) in enumerate(zip(MODELS, SHORT)):
            ax = fig.add_subplot(gs_top[ri, ci])
            base = {a: ACC[(model, "baseline", ed, a)] for a in AXES}
            maa = {a: ACC[(model, "MAA", ed, a)] for a in AXES}
            radar(ax, base, maa, name)
            if ci == 0:
                ax.text(-1.62, 0, f"AJCC {ed}", rotation=90, ha="center", va="center",
                        fontsize=13, weight="bold", clip_on=False)
    fig.text(0.02, 0.955, "A", fontsize=16, weight="bold", va="top")

    # Panels B-E: delta bars with 95% CI + significance (accuracy first, to match Supplementary)
    delta_panel(fig.add_subplot(gs_bot[0, 0]), "acc", "8th", "Accuracy improvement (Δ) — AJCC 8th", "B")
    delta_panel(fig.add_subplot(gs_bot[0, 1]), "acc", "9th", "Accuracy improvement (Δ) — AJCC 9th", "C")
    delta_panel(fig.add_subplot(gs_bot[1, 0]), "f1", "8th", "F1-score improvement (Δ) — AJCC 8th", "D")
    delta_panel(fig.add_subplot(gs_bot[1, 1]), "f1", "9th", "F1-score improvement (Δ) — AJCC 9th", "E")

    fig.legend(handles=[mpatches.Patch(facecolor=COLOR_BASE, edgecolor=COLOR_BASE, alpha=0.4, label="Single prompt"),
                        mpatches.Patch(facecolor=COLOR_MAA, edgecolor=COLOR_MAA, alpha=0.4, label="MAA")],
               loc="upper right", fontsize=11, frameon=False, bbox_to_anchor=(0.99, 0.99))
    out = os.path.join(FIG, "Figure2_main_radar_delta_v5.png")
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
