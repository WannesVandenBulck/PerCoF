"""
Simulated users for PerCoF.

For every rejected applicant, we generate preference profiles in two conditions:
- "random": the user did not see SHAP values, so every actionable feature is
            equally likely to be chosen;
- "shap":   the user saw the SHAP values of their own rejection, so features
            that pushed them towards rejection are more likely to be chosen.

Both conditions use the SAME random numbers (same seed) for everything else:
the number of features, the bounds and the acceptable categories. The only
difference is the probability of picking each feature, so any difference in
results comes from the feature choice alone.

Profiles are built as follows:
- features:   k features (2-4) drawn one by one without replacement; the draw
              order is the willingness ranking (a Plackett-Luce model)
- reluctance: the remaining actionable features, ordered the same way
- direction:  the direction that lowers the risk of default
- bounds:     tight / medium / loose, relative to each feature's MAD
"""

import pickle

import numpy as np
import pandas as pd

from problem import HERE, Preferences, load_setup
from relaxation import IMPROVING_DIRECTION


# Bound settings: numeric bound = multiplier * MAD;
# ordinal / nominal: how many categories the user accepts (None = all)
TIGHTNESS = {
    "tight":  {"mad": 0.5, "categories": 1},
    "medium": {"mad": 1.0, "categories": 2},
    "loose":  {"mad": 2.0, "categories": None},
}


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------

def compute_shap(model, setup):
    """
    SHAP values of every rejected applicant, for the class 'bad credit'.
    Positive value = this feature pushed the applicant towards rejection.
    Returns a DataFrame (applicants x model features).
    """
    import shap

    X = setup["rejected"][setup["model_features"]]
    values = shap.TreeExplainer(model).shap_values(X)
    pos = list(model.classes_).index(1)
    if isinstance(values, list):              # older shap versions: list per class
        values = values[pos]
    else:                                     # newer: (applicants, features, classes)
        values = np.asarray(values)[..., pos]
    return pd.DataFrame(values, index=X.index, columns=setup["model_features"])


# ---------------------------------------------------------------------------
# One simulated user
# ---------------------------------------------------------------------------

def draw_ranking(features, weights, rng):
    """Draw all features one by one, each draw proportional to the weights."""
    features, weights = list(features), np.asarray(weights, dtype=float)
    order = []
    while features:
        p = weights / weights.sum()
        i = rng.choice(len(features), p=p)
        order.append(features.pop(i))
        weights = np.delete(weights, i)
    return order


def feature_weights(actionable, shap_row, lam):
    """
    Probability weights for picking features.
    No SHAP: all equal. With SHAP: a mix of the feature's share of the
    positive SHAP values (lam) and an equal share (1 - lam).
    """
    n = len(actionable)
    uniform = np.full(n, 1.0 / n)
    if shap_row is None:
        return uniform
    push = np.clip(shap_row[actionable].values, 0, None)   # only features that hurt
    if push.sum() == 0:
        return uniform
    return lam * push / push.sum() + (1 - lam) * uniform


def acceptable_categories(setup, f, current, n_accept, rng):
    """Categories the user accepts for categorical feature f."""
    spec = setup["feature_spec"][f]
    targets = [c for c in spec.get("targets", []) if c != current]

    if spec["type"] == "ordinal":
        order = spec["order"]
        # only categories in the improving direction, nearest first
        if current in order:
            pos = order.index(current)
            better = [c for c in order[pos + 1:] if c in targets]
        else:                                   # e.g. "unknown savings": start from the bottom
            better = [c for c in order if c in targets]
        return better if n_accept is None else better[:n_accept]

    # nominal: a random subset of the other categories
    if n_accept is None or n_accept >= len(targets):
        return targets
    return list(rng.choice(targets, size=n_accept, replace=False))


def simulate_user(setup, x, tightness, rng, shap_row=None, lam=0.7, k_range=(2, 4)):
    """Generate one Preferences object for applicant x."""
    spec, stats = setup["feature_spec"], setup["stats"]
    actionable = [f for f, s in spec.items() if s["action"] == "actionable"]
    settings = TIGHTNESS[tightness]

    k = rng.integers(k_range[0], k_range[1] + 1)
    ranking = draw_ranking(actionable, feature_weights(actionable, shap_row, lam), rng)
    selected, reluctance = ranking[:k], ranking[k:]

    directions, bounds, categories = {}, {}, {}
    for f in selected:
        if f in IMPROVING_DIRECTION:
            directions[f] = IMPROVING_DIRECTION[f]
        if spec[f]["type"] == "numeric":
            b = settings["mad"] * stats[f]["mad"]
            bounds[f] = max(np.ceil(b), 1) if spec[f].get("integer", False) else b
        else:
            categories[f] = acceptable_categories(setup, f, x[f], settings["categories"], rng)

    return Preferences(selected=selected, directions=directions, bounds=bounds,
                       categories=categories, reluctance=reluctance)


# ---------------------------------------------------------------------------
# All profiles
# ---------------------------------------------------------------------------

def generate_profiles(model, setup, n_profiles=3, tightness_levels=("tight", "medium", "loose"),
                      lam=0.7, seed=0):
    """
    For every rejected applicant, every profile number and every tightness level,
    one user per condition ("random" and "shap"), generated with the same seed.
    Returns a list of dicts.
    """
    shap_values = compute_shap(model, setup)
    rejected = setup["rejected"]
    profiles = []

    for a, (idx, x) in enumerate(rejected.iterrows()):
        for p in range(n_profiles):
            for t, tight in enumerate(tightness_levels):
                user_seed = seed * 1_000_000 + a * 1_000 + p * 10 + t   # unique, reproducible
                for condition in ("random", "shap"):
                    rng = np.random.default_rng(user_seed)    # same seed in both conditions
                    shap_row = shap_values.loc[idx] if condition == "shap" else None
                    prefs = simulate_user(setup, x, tight, rng, shap_row, lam)
                    profiles.append({
                        "applicant": idx, "profile": p, "tightness": tight,
                        "condition": condition, "seed": user_seed, "prefs": prefs,
                    })
    return profiles, shap_values


if __name__ == "__main__":
    model, setup = load_setup()
    profiles, shap_values = generate_profiles(model, setup)

    with open(HERE / "profiles.pkl", "wb") as f:
        pickle.dump({"profiles": profiles, "shap": shap_values}, f)
    print(f"Generated {len(profiles)} profiles for {len(setup['rejected'])} applicants")

    # Show both conditions for one applicant
    for pr in profiles[:2]:
        prefs = pr["prefs"]
        print(f"\nApplicant {pr['applicant']}, {pr['tightness']}, condition: {pr['condition']}")
        print(f"  willing (ranked): {prefs.selected}")
        print(f"  directions: {prefs.directions}")
        print(f"  bounds: {prefs.bounds}")
        print(f"  categories: {prefs.categories}")
        print(f"  reluctance: {prefs.reluctance}")

    # How strongly does SHAP shift the choice? Share of picks per feature
    picks = pd.DataFrame([{"condition": pr["condition"], "feature": f}
                          for pr in profiles for f in pr["prefs"].selected])
    print("\nShare of users choosing each feature:")
    print((pd.crosstab(picks["feature"], picks["condition"])
           / (len(profiles) / 2)).round(2).sort_values("shap", ascending=False))
