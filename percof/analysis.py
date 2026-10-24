"""
Analysis of the PerCoF experiments: tables and statistical tests.

Reads every results_budget_*.parquet and convergence_budget_*.parquet in this
folder and writes CSV tables to the 'tables' folder, plus a readable summary
(tables/summary.txt).

Key definitions
- gap:       preference cost minus the best preference cost known for the same
             profile and seed (over all methods and budgets), computed only when
             all compared runs end at the same stage. 0 = best known solution.
- stability: mean Jaccard similarity between the sets of changed features over
             the seeds of one profile and method (1 = always the same features).
- Tests are run on APPLICANT level: results are first averaged per applicant
  (44 applicants), because the profiles of one applicant are not independent.
  Methods: Friedman test + Kendall's W, then pairwise Wilcoxon signed-rank
  tests with Holm correction and the matched-pairs rank-biserial correlation.
  SHAP vs random: Wilcoxon signed-rank test per metric (same applicants).

Usage:  python analysis.py
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from problem import HERE

OUT = HERE / "tables"
METHOD_ORDER = ["NSGA-II", "GA", "SA", "Random search"]
STAGE_ORDER = ["original", "relaxed bounds", "added features", "none"]


# ---------------------------------------------------------------------------
# Loading and derived columns
# ---------------------------------------------------------------------------

def load_results():
    frames = [pd.read_parquet(p) for p in sorted(HERE.glob("results_budget_*.parquet"))]
    if not frames:
        raise FileNotFoundError("No results_budget_*.parquet files found")
    r = pd.concat(frames, ignore_index=True)
    r["k_group"] = pd.cut(r["k"], [1, 3, 5, 11], labels=["2-3", "4-5", "6+"])
    return add_gap(r)


def add_gap(r):
    """Gap to the best known preference cost, only where all runs share the stage."""
    key = ["profile_id", "seed"]
    same_stage = r.groupby(key)["stage"].transform("nunique") == 1
    best = r[same_stage].groupby(key)["preference_cost"].transform("min")
    r["gap"] = np.nan
    r.loc[same_stage, "gap"] = r.loc[same_stage, "preference_cost"] - best
    r["is_best"] = r["gap"] < 1e-9
    return r


def jaccard_mean(changed):
    sets = [set(c.split(",")) if c else set() for c in changed]
    pairs = [len(a & b) / len(a | b) if (a | b) else 1.0
             for a, b in itertools.combinations(sets, 2)]
    return np.mean(pairs) if pairs else np.nan


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation (-1..1); positive: x > y."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def holm(pvalues):
    """Holm-Bonferroni correction."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def compare_methods(r, metric, lower_is_better=True):
    """
    Friedman test over methods on applicant-level means, then pairwise Wilcoxon.
    Returns (summary dict, pairwise DataFrame).
    """
    wide = r.groupby(["applicant", "method"])[metric].mean().unstack()
    wide = wide[[m for m in METHOD_ORDER if m in wide]].dropna()
    n, k = wide.shape
    chi2, p = stats.friedmanchisquare(*[wide[m] for m in wide])
    summary = {"metric": metric, "n_applicants": n, "friedman_chi2": chi2,
               "friedman_p": p, "kendalls_w": chi2 / (n * (k - 1))}

    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        diff = wide[a] - wide[b]
        p_w = stats.wilcoxon(wide[a], wide[b]).pvalue if (diff != 0).any() else 1.0
        rows.append({"metric": metric, "method_a": a, "method_b": b,
                     "mean_a": wide[a].mean(), "mean_b": wide[b].mean(),
                     "p": p_w, "rank_biserial": rank_biserial(wide[a], wide[b])})
    pairs = pd.DataFrame(rows)
    pairs["p_holm"] = holm(pairs["p"])
    if lower_is_better:
        pairs["better"] = np.where(pairs["mean_a"] < pairs["mean_b"], pairs["method_a"], pairs["method_b"])
    else:
        pairs["better"] = np.where(pairs["mean_a"] > pairs["mean_b"], pairs["method_a"], pairs["method_b"])
    return summary, pairs


def compare_conditions(r, metrics):
    """SHAP vs random, paired on applicant level, one Wilcoxon test per metric."""
    rows = []
    for metric in metrics:
        wide = r.groupby(["applicant", "condition"])[metric].mean().unstack().dropna()
        diff = wide["shap"] - wide["random"]
        p = stats.wilcoxon(wide["shap"], wide["random"]).pvalue if (diff != 0).any() else 1.0
        rows.append({"metric": metric, "random": wide["random"].mean(),
                     "shap": wide["shap"].mean(), "p": p,
                     "rank_biserial": rank_biserial(wide["shap"], wide["random"])})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)
    r = load_results()
    budgets = sorted(r["budget"].unique())
    lines = [f"Runs: {len(r)}, budgets: {budgets}, profiles: {r['profile_id'].nunique()}, "
             f"applicants: {r['applicant'].nunique()}\n"]

    def save(df, name, title):
        df.to_csv(OUT / f"{name}.csv")
        shown = df.copy()
        for col in shown.columns:
            if str(col) in ("p", "p_holm", "friedman_p"):
                shown[col] = shown[col].map(lambda v: f"{v:.1e}")
        lines.append(f"=== {title} ===\n{shown.round(3).to_string()}\n")

    main_budget = max(budgets)
    rb = r[r["budget"] == main_budget]

    # 1. Reachability: stage by condition and tightness (stages hardly depend on the method)
    t = pd.crosstab([rb["condition"], rb["tightness"]], rb["stage"], normalize="index")
    save(t[[s for s in STAGE_ORDER if s in t]], "stage_by_condition",
         f"Stage reached, by condition and tightness (budget {main_budget}, all methods)")
    t = pd.crosstab(rb["method"], rb["stage"], normalize="index")
    save(t[[s for s in STAGE_ORDER if s in t]], "stage_by_method", "Stage reached, by method")
    same = rb.groupby(["profile_id", "seed"])["stage"].nunique().eq(1).mean()
    lines.append(f"Share of profile-seed combinations where all methods reach the same stage: {same:.3f}\n")

    # 2. Solution quality per method and budget
    q = r.groupby(["budget", "method"]).agg(
        gap=("gap", "mean"), best_share=("is_best", "mean"),
        preference_cost=("preference_cost", "mean"), sparsity=("sparsity", "mean"),
        proximity=("proximity", "mean"), relaxation=("relaxation_amount", "mean"))
    save(q, "quality_by_method", "Solution quality by budget and method")
    save(rb.groupby(["k", "method"])["gap"].mean().unstack()[METHOD_ORDER],
         "gap_by_k", f"Gap by number of selected features k (budget {main_budget})")
    save(rb.groupby(["stage", "method"])["gap"].mean().unstack()[METHOD_ORDER],
         "gap_by_stage", f"Gap by stage (budget {main_budget})")

    # 3. Stability and cost
    stab = (r.groupby(["budget", "profile_id", "method"])["changed"].apply(jaccard_mean)
             .groupby(["budget", "method"]).mean().unstack()[METHOD_ORDER])
    save(stab, "stability", "Stability (Jaccard similarity of changed features across seeds)")
    cost = r.groupby(["budget", "method"])[["runtime", "evaluations", "searches", "first_valid"]].mean()
    save(cost, "cost_by_method", "Computational cost by budget and method")

    # 4. Tests between methods (per budget)
    summaries, pairs = [], []
    for b in budgets:
        for metric in ["gap", "proximity"]:
            s, p = compare_methods(r[r["budget"] == b], metric)
            summaries.append({"budget": b, **s})
            pairs.append(p.assign(budget=b))
    save(pd.DataFrame(summaries).set_index(["budget", "metric"]), "tests_methods_friedman",
         "Friedman tests between methods (applicant-level means)")
    save(pd.concat(pairs).set_index(["budget", "metric", "method_a", "method_b"]),
         "tests_methods_pairwise", "Pairwise Wilcoxon tests (Holm-corrected)")

    # 5. SHAP vs random (main budget, NSGA-II = best method; stages hardly differ by method)
    rs = rb[rb["method"] == "NSGA-II"].copy()
    rs["within_limits"] = rs["stage"] == "original"
    rs["features_added"] = rs["n_added_used"] > 0
    cond = compare_conditions(rs, ["within_limits", "features_added", "preference_cost",
                                   "relaxation_amount", "searches", "sparsity"])
    save(cond.set_index("metric"), "tests_shap_vs_random",
         f"SHAP vs random (NSGA-II, budget {main_budget}, applicant-level, Wilcoxon)")
    save(pd.crosstab([rs["tightness"]], rs["condition"],
                     values=rs["within_limits"], aggfunc="mean"),
         "within_limits_by_tightness", "Share solved within own limits, by tightness")

    # 6. Convergence (first search), per budget
    for b in budgets:
        path = HERE / f"convergence_budget_{b}.parquet"
        if not path.exists():
            continue
        c = pd.read_parquet(path)
        cols = [col for col in c.columns if col[0] == "c" and col[1:].isdigit()]
        final = c[cols[-1]]
        key = ["profile_id", "seed"]
        ok = final.notna().groupby([c["profile_id"], c["seed"]]).transform("all")
        d = c[ok].copy()
        best = d.groupby(key)[cols[-1]].transform("min")
        points = [round(len(cols) * f) - 1 for f in (0.05, 0.1, 0.25, 0.5, 1.0)]
        conv = pd.DataFrame({
            f"gap@{round(b * (i + 1) / len(cols))}": (d[cols[i]] - best).groupby(d["method"]).mean()
            for i in points}).loc[METHOD_ORDER]
        save(conv, f"convergence_gap_budget_{b}",
             f"Gap during the first search (budget {b}), at evaluations")
        valid = pd.DataFrame({
            f"valid@{round(b * (i + 1) / len(cols))}": c[cols[i]].notna().groupby(c["method"]).mean()
            for i in points}).loc[METHOD_ORDER]
        save(valid, f"convergence_valid_budget_{b}",
             f"Share of first searches with a valid solution (budget {b}), at evaluations")
        continue
        save(conv, f"convergence_gap_budget_{b}",
             f"Gap during the first search (budget {b}), at evaluations")

    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Tables saved in {OUT}")


if __name__ == "__main__":
    main()
