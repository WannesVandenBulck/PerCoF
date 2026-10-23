"""
PerCoF: the full procedure for one applicant.

1. Search within the user's own constraints.
2. Bound relaxation: if nothing is found, loosen the bounds of the selected
   features one by one, following the willingness ranking (rank 1 first).
3. Feature addition: if even fully loosened bounds fail, reset the bounds,
   add the first feature of the reluctance ranking and search again
   (then relax again if needed, then add the next reluctant feature, ...).

Any search method from metaheuristics.py can be plugged in.
"""

import copy

import numpy as np

from problem import PerCoFProblem


# Direction that lowers the risk of default, used for features that are added
# from the reluctance ranking (the user gave no direction for those).
# Nominal features have no direction.
IMPROVING_DIRECTION = {
    "duration": "down",
    "amount": "down",
    "installment_rate": "down",
    "number_credits": "down",
    "status": "up",
    "savings": "up",
    "job": "up",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def domain_limit(setup, x, prefs, f):
    """Largest change the data domain allows for numeric feature f in its direction."""
    s = setup["stats"][f]
    direction = prefs.directions.get(f, "both")
    down = x[f] - s["min"] if direction in ("down", "both") else 0
    up = s["max"] - x[f] if direction in ("up", "both") else 0
    return max(down, up, 0)


def all_categories(setup, f):
    """All categories a categorical feature may take as counterfactual value."""
    return list(setup["feature_spec"][f].get("targets", []))


def is_numeric(setup, f):
    return setup["feature_spec"][f]["type"] == "numeric"


class Runner:
    """Runs searches and keeps count of everything spent across all stages."""

    def __init__(self, model, setup, x, method, budget, rng):
        self.model, self.setup, self.x = model, setup, x
        self.method, self.budget, self.rng = method, budget, rng
        self.evaluations = 0
        self.runtime = 0.0
        self.searches = 0
        self.log = []          # result of every search, in order (first = user's own limits)

    def search(self, prefs):
        problem = PerCoFProblem(self.model, self.setup, self.x, prefs)
        result = self.method(problem, self.budget, self.rng)
        self.evaluations += result["evaluations"]
        self.runtime += result["runtime"]
        self.searches += 1
        self.log.append(result)
        return result, problem


# ---------------------------------------------------------------------------
# Stage 1: bound relaxation
# ---------------------------------------------------------------------------

def relax_bounds(runner, prefs, step_frac=1.0, skip=()):
    """
    Loosen bounds following the willingness ranking.

    For each feature (rank 1 first):
      - first try the feature at its domain limit; if even that fails,
        keep it fully loosened and move on to the next feature
        (this skips many pointless searches);
      - otherwise, increase the bound in steps of step_frac * MAD until a
        valid counterfactual is found.
    Categorical features are loosened in one step: all allowed categories.

    Returns (result, problem, prefs) of the first success, or None.
    """
    setup = runner.setup
    prefs = copy.deepcopy(prefs)

    for f in prefs.selected:
        if f in skip:
            continue

        if is_numeric(setup, f):
            limit = domain_limit(setup, runner.x, prefs, f)
            original = prefs.bounds[f]
            if limit <= original:
                continue                      # cannot be loosened any further

            prefs.bounds[f] = limit           # can it work at all?
            res, prob = runner.search(prefs)
            if not res["found"]:
                continue                      # no: keep it loosened, next feature

            step = step_frac * setup["stats"][f]["mad"]
            b = original + step
            while b < limit:                  # yes: find the smallest bound that works
                prefs.bounds[f] = b
                res_b, prob_b = runner.search(prefs)
                if res_b["found"]:
                    return res_b, prob_b, prefs
                b += step
            prefs.bounds[f] = limit
            return res, prob, prefs

        else:
            wanted = set(prefs.categories.get(f, all_categories(setup, f)))
            full = set(all_categories(setup, f))
            if wanted >= full:
                continue
            prefs.categories[f] = sorted(full)
            res, prob = runner.search(prefs)
            if res["found"]:
                return res, prob, prefs

    return None


# ---------------------------------------------------------------------------
# The full PerCoF procedure
# ---------------------------------------------------------------------------

def percof(model, setup, x, prefs, method, budget, rng, step_frac=1.0):
    """
    Run PerCoF for one applicant.

    Returns a report dict:
      stage:       "original", "relaxed bounds", "added features" or "none"
      result:      the search result of the final counterfactual (or None)
      problem:     the PerCoFProblem it was found in (for describe())
      relaxations: {feature: (user's bound, change actually needed)}
      added:       features added from the reluctance ranking (tried)
      added_used:  added features the counterfactual actually changes
      evaluations, runtime, searches: totals over all stages
      log:         the result of every search, in order
    """
    runner = Runner(model, setup, x, method, budget, rng)

    def report(stage, res=None, prob=None, final=None, added=()):
        """
        Relaxations are reported as what the counterfactual ACTUALLY needs beyond
        the user's limits, not as the bound the search happened to use: if the
        search was allowed 14 months but only needed 9, the report says 9.
        """
        relax, used_added = {}, []
        if res is not None:
            cf = prob.decode(res["solution"]).iloc[0]
            for f in prefs.selected:
                if is_numeric(setup, f):
                    change = abs(cf[f] - x[f])
                    if setup["feature_spec"][f].get("integer", False):
                        change = int(round(change))
                    if change > prefs.bounds[f] + 1e-9:
                        relax[f] = (prefs.bounds[f], change)
                else:
                    wanted = prefs.categories.get(f, all_categories(setup, f))
                    if cf[f] != x[f] and cf[f] not in wanted:
                        relax[f] = (wanted, cf[f])
            used_added = [f for f in added if cf[f] != x[f]]
        return {
            "stage": stage, "result": res, "problem": prob,
            "relaxations": relax, "added": list(added), "added_used": used_added,
            "evaluations": runner.evaluations, "runtime": runner.runtime,
            "searches": runner.searches, "log": runner.log,
        }

    # Stage 0: the user's own constraints
    res, prob = runner.search(prefs)
    if res["found"]:
        return report("original", res, prob, prefs)

    # Stage 1: loosen bounds
    out = relax_bounds(runner, prefs, step_frac)
    if out:
        return report("relaxed bounds", *out)

    # Stage 2: add reluctant features one by one
    current = copy.deepcopy(prefs)
    added = []
    for f in prefs.reluctance:
        current = copy.deepcopy(current)
        current.selected.append(f)            # rank = last
        added.append(f)
        if is_numeric(setup, f):
            current.directions[f] = IMPROVING_DIRECTION.get(f, "both")
            current.bounds[f] = domain_limit(setup, x, current, f)
        else:
            if f in IMPROVING_DIRECTION:
                current.directions[f] = IMPROVING_DIRECTION[f]
            current.categories[f] = all_categories(setup, f)

        # the original features keep the user's own bounds
        res, prob = runner.search(current)
        if res["found"]:
            return report("added features", res, prob, current, added)

        out = relax_bounds(runner, current, step_frac, skip=added)
        if out:
            return report("added features", *out, added=added)

    return report("none")


def print_report(rep, x):
    print(f"  Stage: {rep['stage']}")
    for f, (old, new) in rep["relaxations"].items():
        print(f"  Beyond user's limit on {f}: allowed {old}, needed {new}")
    if rep["added"]:
        print(f"  Added features (tried): {', '.join(rep['added'])}; "
              f"used: {', '.join(rep['added_used']) or '-'}")
    if rep["result"]:
        o = rep["result"]["objectives"]
        print(f"  P(bad) {rep['problem'].predict(x[rep['problem'].model_features].values[None, :].astype(float))[0]:.2f}"
              f" -> {o['proba']:.2f}")
        for line in rep["problem"].describe(rep["result"]["solution"]):
            print("    ", line)
    print(f"  Searches: {rep['searches']}, evaluations: {rep['evaluations']}, "
          f"runtime: {rep['runtime']:.2f} s")


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from metaheuristics import run_ga
    from problem import Preferences, load_setup

    model, setup = load_setup()
    budget = 1000

    # Deliberately tight preferences, so the relaxation is needed
    tight = Preferences(
        selected=["duration", "amount"],
        directions={"duration": "down", "amount": "down"},
        bounds={"duration": 2, "amount": 200},
        reluctance=["savings", "status", "other_debtors"],
    )
    # Only one weak feature, so a reluctant feature must be added
    narrow = Preferences(
        selected=["installment_rate"],
        directions={"installment_rate": "down"},
        bounds={"installment_rate": 1},
        reluctance=["savings", "status", "other_debtors"],
    )

    for name, prefs in [("Tight bounds", tight), ("One weak feature", narrow)]:
        print(f"=== {name} ===")
        for idx in range(3):
            x = setup["rejected"].iloc[idx]
            print(f"Applicant {idx}")
            rep = percof(model, setup, x, prefs, run_ga, budget, np.random.default_rng(0))
            print_report(rep, x)
        print()
