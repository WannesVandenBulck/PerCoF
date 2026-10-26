"""
Figures for the PerCoF experiments, in a classic journal style: greys with one
dark-red accent for NSGA-II (the recommended method).

Saves every figure as PDF (vector, for LaTeX) and PNG in 'figures':
1. gap_by_budget:  final gap per budget
2. convergence:    gap to the best solution during the first search
3. gap_by_k:       final gap by number of selected features
4. stages:         stage reached, random vs SHAP, per tightness
5. runtime:        runtime per run, per method and budget

Style: Computer Modern font (as in the LaTeX text), closed axes with inward
ticks, methods told apart by line style and marker as well as shade, so the
figures also remain readable when printed in black and white.

Usage:  python visualise.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from analysis import METHOD_ORDER, load_results
from problem import HERE

OUT = HERE / "figures"
WIDTH = 4.8          # inches = LNCS text width (12.2 cm)
HEIGHT = 2.5

ACCENT = "#8b1a1a"     # dark red
DARK, MID, LIGHT = "#333333", "#888888", "#bdbdbd"

# One style per method: shade + line style + marker
STYLE = {
    "NSGA-II":       dict(color=ACCENT, ls="-",  marker="o", mfc=ACCENT),
    "GA":            dict(color=DARK,   ls="--", marker="s", mfc="white"),
    "SA":            dict(color=DARK,   ls="-.", marker="^", mfc="white"),
    "Random search": dict(color=MID,    ls=":",  marker="x"),
}
BAR_FILL = {"NSGA-II": ACCENT, "GA": DARK, "SA": LIGHT, "Random search": "white"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 0.9, "lines.markersize": 4, "lines.markeredgewidth": 0.8,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
    "legend.framealpha": 1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved figures/{name}.pdf/.png")


def plain_log_axis(ax):
    """Log axis with plain tick labels (0.01, 0.1, 1) instead of 10^-2."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------

def plot_gap_by_budget(r):
    data = r.groupby(["budget", "method"])["gap"].mean().unstack()[METHOD_ORDER]
    x = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    for m in METHOD_ORDER:
        ax.plot(x, data[m].values, label=m, **STYLE[m])
    ax.set_xticks(x, [f"{b:,}" for b in data.index])
    ax.set_xlim(-0.2, len(x) - 0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Budget $B$ (model evaluations per search)")
    ax.set_ylabel("Mean gap to best known solution")
    ax.legend(loc="upper right")
    save(fig, "gap_by_budget")


def plot_convergence(budget):
    c = pd.read_parquet(HERE / f"convergence_budget_{budget}.parquet")
    cols = [col for col in c.columns if col[0] == "c" and col[1:].isdigit()]
    evals = np.linspace(budget / len(cols), budget, len(cols))

    # only runs where every method found a valid candidate in the first search
    ok = c[cols[-1]].notna().groupby([c["profile_id"], c["seed"]]).transform("all")
    d = c[ok]
    best = d.groupby(["profile_id", "seed"])[cols[-1]].transform("min")
    gaps = d[cols].sub(best, axis=0)

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    every = max(len(cols) // 10, 1)
    for m in METHOD_ORDER:
        mean = gaps[d["method"] == m].mean().clip(lower=1e-3)
        ax.plot(evals, mean.values, label=m, markevery=(every // 2, every), **STYLE[m])
    ax.set_yscale("log")
    plain_log_axis(ax)
    ax.set_xlim(0, budget)
    ax.set_xlabel("Model evaluations")
    ax.set_ylabel("Mean gap to best final solution")
    ax.legend(loc="lower left")
    save(fig, "convergence")


def plot_gap_by_k(r, budget):
    rb = r[r["budget"] == budget]
    data = rb.groupby(["k", "method"])["gap"].mean().unstack()[METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    for m in METHOD_ORDER:
        ax.plot(data.index, data[m].values, label=m, **STYLE[m])
    ax.set_xticks(list(data.index))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Number of features selected by the user ($k$)")
    ax.set_ylabel("Mean gap to best known solution")
    ax.legend(loc="upper left")
    save(fig, "gap_by_k")


def plot_stages(r, budget):
    rb = r[(r["budget"] == budget) & (r["method"] == "NSGA-II")]
    stages = ["original", "relaxed bounds", "added features"]
    labels = ["Within own limits", "Bounds relaxed", "Feature added"]
    fills = ["#3a3a3a", LIGHT, ACCENT]
    share = pd.crosstab([rb["tightness"], rb["condition"]], rb["stage"], normalize="index")
    share = share.reindex(columns=stages, fill_value=0)

    order = [(t, c) for t in ["tight", "medium", "loose"] for c in ["random", "shap"]]
    share = share.loc[order] * 100
    x = np.arange(len(order)) + np.repeat([0, 0.5, 1.0], 2)

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT + 0.2))
    bottom = np.zeros(len(order))
    for s, lab, col in zip(stages, labels, fills):
        vals = share[s].values
        ax.bar(x, vals, 0.75, bottom=bottom, label=lab, color=col,
               edgecolor="black", lw=0.6)
        bottom += vals
    ax.set_xticks(x, ["Random" if c == "random" else "SHAP" for _, c in order])
    ax.tick_params(axis="x", length=0)
    for i, t in enumerate(["Tight bounds", "Medium bounds", "Loose bounds"]):
        ax.text(x[2 * i:2 * i + 2].mean(), -0.13, t, ha="center", va="top",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Share of users (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "stages")


def plot_runtime(r):
    data = r.groupby(["budget", "method"])["runtime"].mean().unstack()[METHOD_ORDER] * 1000
    x = np.arange(len(data.index))
    w = 0.2
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    for i, m in enumerate(METHOD_ORDER):
        ax.bar(x + (i - 1.5) * w, data[m].values, w, label=m, color=BAR_FILL[m],
               edgecolor="black", lw=0.6)
    ax.set_yscale("log")
    plain_log_axis(ax)
    ax.set_xticks(x, [f"{b:,}" for b in data.index])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Budget $B$ (model evaluations per search)")
    ax.set_ylabel("Mean runtime per run (ms)")
    ax.legend(loc="upper left")
    save(fig, "runtime")


def main():
    OUT.mkdir(exist_ok=True)
    r = load_results()
    budgets = sorted(r["budget"].unique())
    main_budget = max(budgets)
    print(f"Budgets found: {budgets}; detailed figures for budget {main_budget}")

    if len(budgets) > 1:
        plot_gap_by_budget(r)
    plot_convergence(main_budget)
    plot_gap_by_k(r, main_budget)
    plot_stages(r, main_budget)
    plot_runtime(r)


if __name__ == "__main__":
    main()
