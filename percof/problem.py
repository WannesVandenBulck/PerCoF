"""
PerCoF problem definition.

Contains everything the metaheuristics share:
- Preferences: what one rejected applicant is willing to change
- PerCoFProblem: encoding, repair, decoding and the three objectives

A solution is a vector with one gene per selected feature:
- numeric feature:     the change (delta), e.g. -6 months
- categorical feature: an index into that feature's list of options,
                       where index 0 is always the applicant's current value
A population is a 2D numpy array of shape (n_solutions, n_genes).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent   # the folder this script is in


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_setup(model_path=HERE / "RF.pkl", setup_path=HERE / "percof_setup.pkl"):
    """Load the trained model and the PerCoF setup created in prep_credit.ipynb."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(setup_path, "rb") as f:
        setup = pickle.load(f)
    return model, setup


class FastPredictor:
    """
    Returns P(bad credit) for a numpy array of applicants.

    For a random forest, it averages the trees directly, which gives exactly
    the same result as model.predict_proba but is about 15x faster for small
    batches (predict_proba has a large fixed overhead per call). The
    metaheuristics call the model thousands of times, so this matters.
    Any other model falls back to predict_proba.
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.pos = list(model.classes_).index(1)          # column of class 1 = bad
        self.trees = [e.tree_ for e in getattr(model, "estimators_", [])
                      if hasattr(e, "tree_")]

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.trees:
            total = np.zeros(len(X))
            for t in self.trees:
                p = t.predict(X).reshape(len(X), -1)
                total += p[:, self.pos] / p.sum(axis=1)
            return total / len(self.trees)
        df = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_proba(df)[:, self.pos]


def map_value(setup, feature, value):
    """Return the readable label for an encoded value."""
    return setup["feature_mappings"].get(feature, {}).get(int(value), str(value))


# ---------------------------------------------------------------------------
# User preferences
# ---------------------------------------------------------------------------

@dataclass
class Preferences:
    """
    What one rejected applicant tells PerCoF before the search.

    selected:   features the user is willing to change, ordered from most
                to least willing (so the list order IS the willingness ranking)
    directions: per numeric/ordinal feature: "up", "down" or "both"
    bounds:     per numeric feature: maximum absolute change (e.g. 6 months)
    categories: per categorical feature: acceptable new categories
    reluctance: remaining actionable features, from least to most reluctant
    """
    selected: list
    directions: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    reluctance: list = field(default_factory=list)

    def rank(self, feature):
        """Willingness rank: 1 = most willing."""
        return self.selected.index(feature) + 1


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

class PerCoFProblem:
    """
    One PerCoF search: one rejected applicant x with one set of preferences.

    The metaheuristics only use:
        n_genes, random_population(), repair(), evaluate(), fitness()
    """

    def __init__(self, model, setup, x, prefs, tau=0.5, lam=100.0, mu=0.1):
        self.model = model
        self.spec = setup["feature_spec"]
        self.stats = setup["stats"]
        self.model_features = setup["model_features"]
        self.setup = setup

        self.x = x[self.model_features].copy()   # applicant, in model column order
        self.x_array = self.x.values.astype(float)
        self.col = {f: i for i, f in enumerate(self.model_features)}
        self.predict = FastPredictor(model, self.model_features)
        self.prefs = prefs
        self.tau = tau      # rejection threshold on P(bad credit)
        self.lam = lam      # weight of validity loss in the scalar fitness
        self.mu = mu        # weight of sparsity in the scalar fitness

        self.n_evals = 0    # number of model calls (the search budget)

        self._check_preferences()
        self._build_genes()

    # ----- setup -----------------------------------------------------------

    def _check_preferences(self):
        for f in self.prefs.selected:
            if self.spec[f]["action"] != "actionable":
                raise ValueError(f"'{f}' is not actionable")

    def _build_genes(self):
        """
        For every selected feature, compute what values its gene may take.
        Numeric:     lower and upper limit of the change (delta)
        Categorical: list of options; option 0 is the current value
        """
        self.genes = list(self.prefs.selected)
        self.n_genes = len(self.genes)
        self.is_numeric = np.array([self.spec[f]["type"] == "numeric" for f in self.genes])
        self.low = np.zeros(self.n_genes)
        self.high = np.zeros(self.n_genes)
        self.options = {}

        for i, f in enumerate(self.genes):
            current = self.x[f]
            if self.is_numeric[i]:
                self.low[i], self.high[i] = self._numeric_limits(f, current)
            else:
                opts = self._categorical_options(f, current)
                self.options[f] = opts
                self.low[i], self.high[i] = 0, len(opts) - 1

    def _numeric_limits(self, f, current):
        """Limits of the change: user bound and direction, within the data domain."""
        b = self.prefs.bounds[f]
        direction = self.prefs.directions.get(f, "both")
        low = -b if direction in ("down", "both") else 0.0
        high = b if direction in ("up", "both") else 0.0

        # Domain: never leave the range seen in the training data
        dom_min = min(self.stats[f]["min"], current)
        dom_max = max(self.stats[f]["max"], current)
        low = max(low, dom_min - current)
        high = min(high, dom_max - current)
        return low, high

    def _categorical_options(self, f, current):
        """
        Allowed categories: the user's acceptable categories, restricted to the
        targets allowed by the feature specification and, for ordinal features,
        to the allowed direction. The current value always comes first.
        """
        spec = self.spec[f]
        allowed = set(spec.get("targets", []))
        wanted = set(self.prefs.categories.get(f, allowed))
        candidates = allowed & wanted

        if spec["type"] == "ordinal" and current in spec["order"]:
            pos = spec["order"].index(current)
            direction = self.prefs.directions.get(f, "both")
            if direction == "up":
                candidates = {c for c in candidates if spec["order"].index(c) > pos}
            elif direction == "down":
                candidates = {c for c in candidates if spec["order"].index(c) < pos}
        # (if the current value is outside the order, e.g. "unknown savings",
        #  any allowed target counts as a move)

        candidates.discard(current)
        return [current] + sorted(candidates)

    # ----- population, repair, decoding ------------------------------------

    def random_population(self, n, rng):
        """n random feasible solutions; the first one is 'change nothing'."""
        pop = rng.uniform(self.low, self.high, size=(n, self.n_genes))
        pop[0] = 0.0
        return self.repair(pop)

    def repair(self, pop):
        """Push every gene back within its limits and round where needed."""
        pop = np.clip(np.atleast_2d(pop).astype(float), self.low, self.high)
        for i, f in enumerate(self.genes):
            if not self.is_numeric[i] or self.spec[f].get("integer", False):
                pop[:, i] = np.round(pop[:, i])
        # rounding can step just outside the limits, so clip once more
        return np.clip(pop, self.low, self.high)

    def decode_array(self, pop):
        """Turn a population of genes into counterfactual applicants (numpy array)."""
        pop = np.atleast_2d(pop)
        cf = np.tile(self.x_array, (len(pop), 1))
        for i, f in enumerate(self.genes):
            j = self.col[f]
            if self.is_numeric[i]:
                cf[:, j] = self.x_array[j] + pop[:, i]
            else:
                cf[:, j] = np.array(self.options[f], dtype=float)[pop[:, i].astype(int)]
        return cf

    def decode(self, pop):
        """Same as decode_array, but as a readable DataFrame."""
        return pd.DataFrame(self.decode_array(pop), columns=self.model_features)

    # ----- objectives ------------------------------------------------------

    def evaluate(self, pop):
        """
        Compute the three objectives for a whole population in one model call.
        Returns a dict with arrays: V (validity loss), P (preference cost),
        N (sparsity), proba (predicted P(bad credit)), valid (bool).
        """
        pop = np.atleast_2d(pop)
        proba = self.predict(self.decode_array(pop))
        self.n_evals += len(pop)

        changed = np.zeros_like(pop, dtype=bool)
        cost = np.zeros(len(pop))
        for i, f in enumerate(self.genes):
            if self.is_numeric[i]:
                changed[:, i] = pop[:, i] != 0
                c = np.abs(pop[:, i]) / self.stats[f]["mad"]
            else:
                changed[:, i] = pop[:, i] != 0      # option 0 = current value
                c = changed[:, i].astype(float)
            cost += self.prefs.rank(f) * c

        V = np.maximum(0.0, proba - self.tau)
        # V is 0 when proba is just below tau; validity itself is proba < tau
        return {
            "V": V,
            "P": cost,
            "N": changed.sum(axis=1),
            "proba": proba,
            "valid": proba < self.tau,
        }

    def fitness(self, obj):
        """Scalar fitness F = lam*V + P + mu*N (lower is better), for GA and SA.
        Invalid solutions with V = 0 (proba exactly tau) get a small penalty."""
        V = obj["V"] + (~obj["valid"]) * 1e-3
        return self.lam * V + obj["P"] + self.mu * obj["N"]

    # ----- reporting -------------------------------------------------------

    def describe(self, solution):
        """Readable list of the changes in one solution."""
        cf = self.decode(solution).iloc[0]
        lines = []
        for f in self.genes:
            if cf[f] != self.x[f]:
                lines.append(f"{self.setup['feature_desc'].get(f, f)}: "
                             f"{self._label(f, self.x[f])} -> {self._label(f, cf[f])}")
        return lines or ["no changes"]

    def _label(self, f, value):
        if self._num(f):
            return int(value) if self.spec[f].get("integer", False) else round(value, 2)
        return map_value(self.setup, f, value)

    def _num(self, f):
        return self.spec[f]["type"] == "numeric"


# ---------------------------------------------------------------------------
# Quick check: random search on one applicant
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, setup = load_setup()
    x = setup["rejected"].iloc[1]

    prefs = Preferences(
        selected=["duration", "amount", "savings"],          # most -> least willing
        directions={"duration": "down", "amount": "down", "savings": "up"},
        bounds={"duration": 12, "amount": 1000},
        reluctance=["housing", "other_debtors"],
    )

    problem = PerCoFProblem(model, setup, x, prefs)
    rng = np.random.default_rng(0)

    pop = problem.random_population(200, rng)
    obj = problem.evaluate(pop)
    F = problem.fitness(obj)

    print(f"Original P(bad credit): {problem.predict(problem.x_array[None, :])[0]:.2f}")
    print(f"Valid solutions found: {obj['valid'].sum()} of {len(pop)}")
    best = np.argmin(F)
    print(f"Best: P(bad) = {obj['proba'][best]:.2f}, preference cost = {obj['P'][best]:.2f}")
    for line in problem.describe(pop[best]):
        print("  ", line)
    print(f"Model evaluations used: {problem.n_evals}")
