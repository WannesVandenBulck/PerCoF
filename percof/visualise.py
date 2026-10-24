"""
Figures for the PerCoF experiments.

Saves every figure as PDF (for LaTeX) and PNG (for a quick look) in 'figures':
1. convergence:     gap to the best solution during the first search
2. gap_by_k:        final gap by number of selected features
3. stages:          stage reached, SHAP vs random, per tightness
4. gap_by_budget:   final gap per budget (only if several budgets were run)
5. runtime:         runtime per PerCoF run, per method

Sized for the LNCS text width (12.2 cm).

Usage:  python visualise.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import METHOD_ORDER, load_results
from problem import HERE

OUT = HERE / "figures"
WIDTH = 4.8                                    # inches, LNCS text width
COLORS = {"NSGA-II": "#1b6ca8", "GA": "#2a9d8f", "SA": "#e76f51", "Random search": "#8d8d8d"}
MARKERS = {"NSGA-II": "o", "GA": "s", "SA": "^", "Random search": "x"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved figures/{name}.pdf/.png")


# ---------------------------------------------------------------------------

def plot_convergence(budget):
    """Mean gap to the best final solution during the first search."""
    c = pd.read_parquet(HERE / f"convergence_budget_{budget}.parquet")
    cols = [col for col in c.columns if col[0] == "c" and col[1:].isdigit()]
    evals = np.linspace(budget / len(cols), budget, len(cols))

    # only runs where every method found a valid solution in the first search
    ok = c[cols[-1]].notna().groupby([c["profile_id"], c["seed"]]).transform("all")
    d = c[ok]
    best = d.groupby(["profile_id", "seed"])[cols[-1]].transform("min")
    gaps = d[cols].sub(best, axis=0)

    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for m in METHOD_ORDER:
        mean = gaps[d["method"] == m].mean()
        ax.plot(evals, mean.clip(lower=1e-3), label=m, color=COLORS[m], lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Model evaluations")
    ax.set_ylabel("Mean gap to best solution")
    ax.legend(frameon=False)
    save(fig, "convergence")


def plot_gap_by_k(r, budget):
    data = r[r["budget"] == budget].groupby(["k", "method"])["gap"].mean().unstack()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for m in METHOD_ORDER:
        ax.plot(data.index, data[m], marker=MARKERS[m], color=COLORS[m], label=m, lw=1.2, ms=4)
    ax.set_xlabel("Number of features selected by the user ($k$)")
    ax.set_ylabel("Mean gap to best solution")
    ax.legend(frameon=False)
    save(fig, "gap_by_k")


def plot_stages(r, budget):
    """Stacked bars: share of users per stage, random vs SHAP, per tightness."""
    rb = r[(r["budget"] == budget) & (r["method"] == "NSGA-II")]
    stages = ["original", "relaxed bounds", "added features"]
    labels = ["Within own limits", "Bounds relaxed", "Features added"]
    shades = ["#1b6ca8", "#8ecae6", "#e76f51"]
    share = pd.crosstab([rb["tightness"], rb["condition"]], rb["stage"], normalize="index")
    share = share.reindex(columns=stages, fill_value=0)

    order = [(t, c) for t in ["tight", "medium", "loose"] for c in ["random", "shap"]]
    share = share.loc[order]
    x = np.arange(len(order)) + np.repeat([0, 0.4, 0.8], 2)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    bottom = np.zeros(len(order))
    for s, lab, col in zip(stages, labels, shades):
        ax.bar(x, share[s], 0.8, bottom=bottom, label=lab, color=col)
        bottom += share[s].values
    ax.set_xticks(x, [("SHAP" if c == "shap" else "Random") for _, c in order])
    for i, t in enumerate(["Tight", "Medium", "Loose"]):
        ax.text(x[2 * i:2 * i + 2].mean(), -0.17, t, ha="center", transform=ax.get_xaxis_transform())
    ax.set_ylabel("Share of users")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, "stages")


def plot_gap_by_budget(r):
    data = r.groupby(["budget", "method"])["gap"].mean().unstack()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for m in METHOD_ORDER:
        ax.plot(data.index, data[m], marker=MARKERS[m], color=COLORS[m], label=m, lw=1.2, ms=4)
    ax.set_xscale("log")
    ax.set_xticks(data.index, [str(b) for b in data.index])
    ax.set_xlabel("Budget per search (model evaluations)")
    ax.set_ylabel("Mean gap to best solution")
    ax.legend(frameon=False)
    save(fig, "gap_by_budget")


def plot_runtime(r, budget):
    rb = r[r["budget"] == budget]
    means = rb.groupby("method")["runtime"].mean().reindex(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    ax.barh(means.index, means.values, color=[COLORS[m] for m in means.index])
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Mean runtime per PerCoF run (s, log scale)")
    for i, v in enumerate(means.values):
        ax.text(v * 1.1, i, f"{v:.2f} s", va="center", fontsize=7)
    save(fig, "runtime")


def main():
    OUT.mkdir(exist_ok=True)
    r = load_results()
    budgets = sorted(r["budget"].unique())
    main_budget = max(budgets)
    print(f"Budgets found: {budgets}; detailed figures for budget {main_budget}")

    plot_convergence(main_budget)
    plot_gap_by_k(r, main_budget)
    plot_stages(r, main_budget)
    plot_runtime(r, main_budget)
    if len(budgets) > 1:
        plot_gap_by_budget(r)
    else:
        print("  (only one budget found: gap_by_budget skipped)")


if __name__ == "__main__":
    main()
