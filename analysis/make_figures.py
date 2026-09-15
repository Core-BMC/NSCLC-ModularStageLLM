"""Build manuscript Figure 2 (performance) and Figure 3 (Stage confusion, MAA 9th).

Reads results/table4.csv, results/table4_f1.csv, results/confusion/*.csv (from
analyze_results.py). Outputs hi-res PNGs to results/figures/.
Run: python3 make_figures.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
CONF = os.path.join(RES, "confusion")
os.makedirs(FIG, exist_ok=True)

MODELS = ["GPT-4o", "LLaMA 3.3 70B", "Phi-4 14B"]
MODES = ["baseline", "MAA"]
MODE_COLOR = {"baseline": "#9ecae1", "MAA": "#08519c"}


def load(fn):
    with open(os.path.join(RES, fn)) as f:
        return list(csv.DictReader(f))


def fig2():
    import matplotlib.patches as mpatches
    acc = {(r["model"], r["mode"], r["edition"], r["axis"]): (float(r["accuracy"]), float(r["ci_lo"]), float(r["ci_hi"]))
           for r in load("table4.csv")}
    f1 = {(r["model"], r["mode"], r["edition"], r["axis"]): (float(r["macro_f1"]), float(r["ci_lo"]), float(r["ci_hi"]))
          for r in load("table4_f1.csv")}
    sig = {(r["model"], r["edition"], r["axis"]): (float(r["acc_p"]), float(r["f1_p"])) for r in load("sig.csv")}
    comps = [("T", "cT"), ("N", "cN"), ("M", "cM")]
    ACC = {"baseline": "#9ecae1", "MAA": "#08519c"}   # blues
    F1C = {"baseline": "#fdae6b", "MAA": "#e6550d"}    # oranges
    # rows: (label, edition, data, metric-key, palette)
    rowspec = [("Accuracy (%)\nAJCC 8th", "8", acc, "acc", ACC),
               ("Accuracy (%)\nAJCC 9th", "9", acc, "acc", ACC),
               ("Macro-F1 (%)\nAJCC 8th", "8", f1, "f1", F1C),
               ("Macro-F1 (%)\nAJCC 9th", "9", f1, "f1", F1C)]
    SHORT = ["GPT-4o", "LLaMA 70B", "Phi-4"]

    def star(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    fig, axes = plt.subplots(4, 3, figsize=(13, 14), sharey=True)
    x = np.arange(len(MODELS)); w = 0.38
    for ri, (rlabel, ed, data, metric, pal) in enumerate(rowspec):
        for ci, (ax_code, ax_title) in enumerate(comps):
            ax = axes[ri][ci]
            tops = []
            for mi, mode in enumerate(MODES):
                vals, los, his = [], [], []
                for model in MODELS:
                    v, lo, hi = data[(model, mode, ed + "th", ax_code)]
                    vals.append(v); los.append(v - lo); his.append(hi - v)
                ax.bar(x + (mi - 0.5) * w, vals, w, color=pal[mode],
                       yerr=[los, his], capsize=3, edgecolor="black", linewidth=0.5)
                tops.append([v + h for v, h in zip(vals, his)])
            for k, model in enumerate(MODELS):  # baseline-vs-MAA comparison bracket per model
                p = sig[(model, ed + "th", ax_code)][0 if metric == "acc" else 1]
                x0, x1 = x[k] - 0.5 * w, x[k] + 0.5 * w      # baseline bar, MAA bar centers
                yb = max(tops[0][k], tops[1][k]) + 2.5
                ax.plot([x0, x0, x1, x1], [yb - 1.2, yb, yb, yb - 1.2], lw=0.8, color="black")
                ax.text(x[k], yb + 0.4, star(p), ha="center", va="bottom", fontsize=8,
                        fontweight="bold" if p < 0.05 else "normal")
            ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=8)
            ax.set_ylim(0, 112); ax.set_yticks([0, 20, 40, 60, 80, 100])
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=9)
            if ri == 0:
                ax.set_title(ax_title, fontsize=11)
            ax.grid(axis="y", alpha=0.3)
    axes[0][2].legend(handles=[mpatches.Patch(color=ACC["baseline"], label="single prompt"),
                               mpatches.Patch(color=ACC["MAA"], label="MAA")], fontsize=9, loc="lower right")
    axes[2][2].legend(handles=[mpatches.Patch(color=F1C["baseline"], label="single prompt"),
                               mpatches.Patch(color=F1C["MAA"], label="MAA")], fontsize=9, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG, "Figure2_final3.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print("wrote", out)


def load_conf(tag):
    """Return (mat, y_labels, x_labels) with the True axis reversed (high stage on top,
    stage 0 at the bottom) to match the original submission, and the non-class reference
    rows dropped ('Unknown' and '(no label)' are not reference classes) while both are
    kept on the Predicted axis.

    The two are different events and the figure must not merge them. '(empty)' in the
    confusion files is the output no parse path could read - the deposited matrices call
    it '(no label)' - whereas 'Unknown' is a stage group that could not be derived from
    an indeterminate or umbrella component. Relabelling '(empty)' to 'Unknown' put the
    one no-label case into the Unknown column of Figure 3, against the manuscript's own
    definition."""
    path = os.path.join(CONF, tag + ".csv")
    with open(path) as f:
        rows = list(csv.reader(f))
    relabel = lambda s: "(no label)" if s in ("(empty)", "") else s
    x_labels = [relabel(x) for x in rows[0][1:]]                      # Predicted (ascending, keeps Unknown)
    row_labels = [relabel(r[0]) for r in rows[1:]]                    # True (ascending, union)
    mat_all = np.array([[int(x) for x in r[1:]] for r in rows[1:]], dtype=int)
    keep = [i for i, l in enumerate(row_labels)
            if l not in ("Unknown", "(no label)")]                     # not reference classes
    mat = mat_all[keep][::-1]                                         # reverse rows -> high stage on top
    y_labels = [row_labels[i] for i in keep][::-1]
    return mat, y_labels, x_labels


def fig3():
    models = [("gpt4o", "GPT-4o"), ("llama3-70b", "LLaMA 3.3 70B"), ("phi4", "Phi-4 14B")]
    rows = [("8", "base", "AJCC 8th · Single prompt"), ("8", "mma", "AJCC 8th · MAA"),
            ("9", "base", "AJCC 9th · Single prompt"), ("9", "mma", "AJCC 9th · MAA")]
    fig, axes = plt.subplots(4, 3, figsize=(16, 21))
    for ri, (edc, mode, rlab) in enumerate(rows):
        for ci, (mkey, mlab) in enumerate(models):
            ax = axes[ri][ci]
            mat, ylabels, xlabels = load_conf(f"{mkey}_{mode}_ajcc{edc}_S")
            rowsum = mat.sum(1, keepdims=True); norm = np.divide(mat, np.where(rowsum == 0, 1, rowsum))
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=6)
            ax.set_xlabel("Predicted stage", fontsize=7.5)
            ax.set_ylabel(("%s\nReference stage" % rlab) if ci == 0 else "Reference stage",
                          fontsize=10 if ci == 0 else 7.5, weight="bold" if ci == 0 else "normal")
            if ri == 0:
                ax.set_title(mlab, fontsize=12, weight="bold")
            for i in range(len(ylabels)):
                for j in range(len(xlabels)):
                    if mat[i, j]:  # annotate row-normalized recall (%), matching the original submission
                        pct = ("%.1f" % (norm[i, j] * 100)).rstrip("0").rstrip(".")
                        ax.text(j, i, pct, ha="center", va="center", fontsize=4.5,
                                color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    out = os.path.join(FIG, "Figure3_confusion_stage_base_MAA_8th_9th.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig2()
    fig3()
