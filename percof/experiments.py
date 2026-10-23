"""
Experiment runner for PerCoF.

Runs every search method on every simulated user profile with several seeds,
and saves:
- results.parquet:     one row per run (profile x method x seed) with all metrics
- convergence.parquet: one row per run with the convergence curve of the FIRST
                       search (the user's own limits, identical problem for every
                       method): best valid F after every 1% of the budget

Results are saved per profile in a 'runs' folder while running, so an
interrupted experiment can simply be restarted: finished profiles are skipped.

Usage (from the folder with the code):
    python experiments.py --limit 6              # quick test on 6 profiles
    python experiments.py --workers 4            # everything, 4 processes
"""

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from metaheuristics import METHODS
from problem import HERE, load_setup
from relaxation import percof

N_POINTS = 100   # points on each convergence curve


# ---------------------------------------------------------------------------
# Metrics for one PerCoF run
# ---------------------------------------------------------------------------

def run_metrics(rep, x, setup):
    """Turn a PerCoF report into one row of metrics."""
    spec, stats = setup["feature_spec"], setup["stats"]
    row = {
        "stage": rep["stage"],                       # original / relaxed bounds / added features / none
        "found": rep["result"] is not None,
        "evaluations": rep["evaluations"],           # total over all searches
        "runtime": rep["runtime"],
        "searches": rep["searches"],
        "n_added": len(rep["added"]),
        "n_added_used": len(rep["added_used"]),
        "added_used": ",".join(rep["added_used"]),
        "n_relaxed": len(rep["relaxations"]),
    }

    # How far beyond the user's limits, in MAD units (numeric) + count (categorical)
    amount = 0.0
    for f, (allowed, needed) in rep["relaxations"].items():
        if spec[f]["type"] == "numeric":
            amount += (needed - allowed) / stats[f]["mad"]
        else:
            amount += 1
    row["relaxation_amount"] = amount

    first = rep["log"][0]
    row["first_valid"] = first["first_valid"]         # evaluations until first valid (own limits)

    if rep["result"] is None:
        return row

    obj = rep["result"]["objectives"]
    cf = rep["problem"].decode(rep["result"]["solution"]).iloc[0]
    changed = [f for f in setup["model_features"] if cf[f] != x[f]]

    # Proximity: unweighted distance to the applicant
    # numeric: |change| / MAD, categorical: 1 per changed feature
    proximity = sum(abs(cf[f] - x[f]) / stats[f]["mad"] if spec[f]["type"] == "numeric" else 1.0
                    for f in changed)

    row.update({
        "preference_cost": obj["P"],
        "sparsity": int(obj["N"]),
        "proximity": proximity,
        "proba_after": obj["proba"],
        "F": rep["result"]["F"],
        "changed": ",".join(sorted(changed)),          # for stability (Jaccard) in the analysis
        "counterfactual": json.dumps({f: float(cf[f]) for f in changed}),
    })
    return row


def convergence_curve(result, budget):
    """Best valid F after every 1% of the budget (NaN = no valid solution yet)."""
    grid = np.linspace(budget / N_POINTS, budget, N_POINTS)
    hist = np.array(result["history"])                 # columns: evaluations used, best F
    curve = np.full(N_POINTS, np.nan)
    for k, g in enumerate(grid):
        seen = hist[hist[:, 0] <= g]
        if len(seen) and np.isfinite(seen[-1, 1]):
            curve[k] = seen[-1, 1]
    return curve


# ---------------------------------------------------------------------------
# One profile: all methods x all seeds
# ---------------------------------------------------------------------------

_G = {}   # data loaded once per worker process


def _init_worker():
    model, setup = load_setup()
    with open(HERE / "profiles.pkl", "rb") as f:
        profiles = pickle.load(f)["profiles"]
    _G.update(model=model, setup=setup, profiles=profiles)


def run_profile(pid, methods, n_seeds, budget):
    """Run every method with every seed on profile pid. Returns (rows, curves)."""
    model, setup = _G["model"], _G["setup"]
    pr = _G["profiles"][pid]
    x = setup["rejected"].loc[pr["applicant"]]
    rows, curves = [], []

    for method in methods:
        for s in range(n_seeds):
            rng = np.random.default_rng(pid * 100 + s)   # same seeds for every method
            rep = percof(model, setup, x, pr["prefs"], METHODS[method], budget, rng)

            keys = {"profile_id": pid, "applicant": pr["applicant"], "profile": pr["profile"],
                    "tightness": pr["tightness"], "condition": pr["condition"],
                    "k": len(pr["prefs"].selected), "method": method, "seed": s,
                    "budget": budget}
            rows.append({**keys, **run_metrics(rep, x, setup)})

            curve = convergence_curve(rep["log"][0], budget)
            curves.append({**keys, **{f"c{k:03d}": v for k, v in enumerate(curve)}})
    return rows, curves


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1000, help="model evaluations per search")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--limit", type=int, default=None, help="only the first N profiles (testing)")
    args = ap.parse_args()

    run_dir = HERE / "runs" / f"budget_{args.budget}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(HERE / "profiles.pkl", "rb") as f:
        n_profiles = len(pickle.load(f)["profiles"])
    todo = [p for p in range(n_profiles if args.limit is None else args.limit)
            if not (run_dir / f"profile_{p:04d}.pkl").exists()]
    print(f"{len(todo)} profiles to run, {len(args.methods)} methods, {args.seeds} seeds, "
          f"budget {args.budget}, {args.workers} worker(s)")

    def save(pid, out):
        with open(run_dir / f"profile_{pid:04d}.pkl", "wb") as f:
            pickle.dump(out, f)

    start, done = time.time(), 0
    if args.workers == 1:
        _init_worker()
        for pid in todo:
            save(pid, run_profile(pid, args.methods, args.seeds, args.budget))
            done += 1
            _progress(done, len(todo), start)
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
            futures = {pool.submit(run_profile, pid, args.methods, args.seeds, args.budget): pid
                       for pid in todo}
            for fut in as_completed(futures):
                save(futures[fut], fut.result())
                done += 1
                _progress(done, len(todo), start)

    combine(run_dir, args.budget)


def _progress(done, total, start):
    elapsed = time.time() - start
    eta = elapsed / done * (total - done)
    print(f"\r  {done}/{total} profiles, {elapsed / 60:.1f} min elapsed, "
          f"~{eta / 60:.1f} min left", end="", flush=True)


def combine(run_dir, budget):
    """Merge all per-profile files into results and convergence tables."""
    rows, curves = [], []
    for path in sorted(run_dir.glob("profile_*.pkl")):
        with open(path, "rb") as f:
            r, c = pickle.load(f)
        rows += r
        curves += c
    results = pd.DataFrame(rows)
    results.to_parquet(HERE / f"results_budget_{budget}.parquet")
    pd.DataFrame(curves).to_parquet(HERE / f"convergence_budget_{budget}.parquet")
    print(f"\nSaved {len(results)} runs to results_budget_{budget}.parquet "
          f"and convergence_budget_{budget}.parquet")


if __name__ == "__main__":
    main()
