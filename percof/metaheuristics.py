"""
Search methods for PerCoF: random search (baseline), GA, NSGA-II and SA.

All four:
- use the same building blocks from problem.py (random_population, repair,
  evaluate, fitness),
- get the same budget of model evaluations,
- return the same kind of result, using the same selection rule:
  the best VALID solution found during the whole search, i.e. the valid
  solution with the lowest F = P + mu*N (for valid solutions V = 0).

Each run_* function returns a dict, see Tracker.result().
"""

import time

import numpy as np


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------

class Tracker:
    """Keeps the best valid solution and a convergence history during a search."""

    def __init__(self, problem, method):
        self.problem = problem
        self.method = method
        self.start_evals = problem.n_evals
        self.start_time = time.perf_counter()
        self.best = None            # genes of best valid solution
        self.best_F = np.inf
        self.best_obj = None
        self.first_valid = None     # evaluations needed to find the first valid solution
        self.history = []           # (evaluations used, best valid F so far)

    def used(self):
        return self.problem.n_evals - self.start_evals

    def update(self, pop, obj, F):
        valid = obj["valid"]
        if valid.any():
            if self.first_valid is None:
                self.first_valid = self.used()
            i = np.flatnonzero(valid)[np.argmin(F[valid])]
            if F[i] < self.best_F:
                self.best_F = F[i]
                self.best = pop[i].copy()
                self.best_obj = {k: v[i] for k, v in obj.items()}
        self.history.append((self.used(), self.best_F))

    def result(self):
        return {
            "method": self.method,
            "found": self.best is not None,
            "solution": self.best,
            "F": self.best_F,
            "objectives": self.best_obj,
            "evaluations": self.used(),
            "first_valid": self.first_valid,
            "runtime": time.perf_counter() - self.start_time,
            "history": self.history,
        }


def mutate(pop, problem, rng, p_mut, sigma):
    """
    Change each gene with probability p_mut.
    Numeric: add Gaussian noise, scaled to the gene's allowed range.
    Categorical: jump to a random allowed option.
    """
    pop = pop.copy()
    span = problem.high - problem.low
    mask = rng.random(pop.shape) < p_mut
    noise = rng.normal(0.0, sigma * np.maximum(span, 1e-9), size=pop.shape)
    random_cat = rng.integers(problem.low, problem.high + 1, size=pop.shape)
    num = problem.is_numeric
    pop[:, num] = np.where(mask[:, num], pop[:, num] + noise[:, num], pop[:, num])
    pop[:, ~num] = np.where(mask[:, ~num], random_cat[:, ~num], pop[:, ~num])
    return problem.repair(pop)


def uniform_crossover(p1, p2, rng, p_cx):
    """Each gene comes from either parent with equal probability."""
    c1, c2 = p1.copy(), p2.copy()
    do = rng.random(len(p1)) < p_cx                      # which pairs cross over
    swap = (rng.random(p1.shape) < 0.5) & do[:, None]
    c1[swap], c2[swap] = p2[swap], p1[swap]
    return np.vstack([c1, c2])


def tournament(scores, n, rng, size=2):
    """Pick n parents; each is the best (lowest score) of `size` random candidates."""
    idx = rng.integers(0, len(scores), size=(n, size))
    return idx[np.arange(n), np.argmin(scores[idx], axis=1)]


# ---------------------------------------------------------------------------
# Random search (baseline)
# ---------------------------------------------------------------------------

def run_random_search(problem, budget, rng, batch=50):
    """Generate random feasible solutions until the budget is used; keep the best."""
    tr = Tracker(problem, "Random search")
    first = True
    while tr.used() < budget:
        n = min(batch, budget - tr.used())
        pop = problem.random_population(n, rng) if first else \
            problem.repair(rng.uniform(problem.low, problem.high, size=(n, problem.n_genes)))
        first = False
        obj = problem.evaluate(pop)
        tr.update(pop, obj, problem.fitness(obj))
    return tr.result()


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

def run_ga(problem, budget, rng, pop_size=40, p_cx=0.9, p_mut=None,
           sigma=0.2, elite=2):
    """
    Generational GA on the scalar fitness F.
    Tournament selection, uniform crossover, Gaussian/random mutation, elitism.
    """
    tr = Tracker(problem, "GA")
    p_mut = p_mut or 1.0 / problem.n_genes

    pop = problem.random_population(pop_size, rng)
    obj = problem.evaluate(pop)
    F = problem.fitness(obj)
    tr.update(pop, obj, F)

    while tr.used() < budget:
        n_off = min(pop_size - elite, budget - tr.used())
        parents = tournament(F, n_off + n_off % 2, rng)
        children = uniform_crossover(pop[parents[0::2]], pop[parents[1::2]], rng, p_cx)
        children = mutate(children[:n_off], problem, rng, p_mut, sigma)

        obj_c = problem.evaluate(children)
        F_c = problem.fitness(obj_c)
        tr.update(children, obj_c, F_c)

        # next generation: the elite of the old population + all children,
        # topped up with the next best old solutions if the budget cut children short
        order = np.argsort(F)
        keep = order[:pop_size - len(children)]
        pop = np.vstack([pop[keep], children])
        F = np.concatenate([F[keep], F_c])
    return tr.result()


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------

def objectives_matrix(obj):
    """The three objectives, all minimised. Invalid solutions with V = 0
    (probability exactly tau) get a tiny penalty, as in problem.fitness."""
    V = obj["V"] + (~obj["valid"]) * 1e-3
    return np.column_stack([V, obj["P"], obj["N"]])


def non_dominated_sort(Y):
    """Return the Pareto front number of each row of Y (0 = best front)."""
    n = len(Y)
    better_eq = (Y[:, None, :] <= Y[None, :, :]).all(axis=2)
    strictly = (Y[:, None, :] < Y[None, :, :]).any(axis=2)
    dominates = better_eq & strictly                  # dominates[i, j]: i dominates j
    dominated_by = dominates.sum(axis=0)
    rank = np.full(n, -1)
    front = 0
    current = np.flatnonzero(dominated_by == 0)
    while len(current):
        rank[current] = front
        dominated_by = dominated_by - dominates[current].sum(axis=0)
        dominated_by[rank >= 0] = -1
        current = np.flatnonzero(dominated_by == 0)
        front += 1
    return rank


def crowding_distance(Y):
    """Larger = more isolated within its front = preferred, to keep diversity."""
    n, m = Y.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(Y[:, k])
        span = Y[order[-1], k] - Y[order[0], k]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0:
            d[order[1:-1]] += (Y[order[2:], k] - Y[order[:-2], k]) / span
    return d


def select_nsga(Y, n):
    """Pick n survivors by front, then by crowding distance."""
    rank = non_dominated_sort(Y)
    crowd = np.zeros(len(Y))
    for r in np.unique(rank):
        members = rank == r
        crowd[members] = crowding_distance(Y[members])
    order = np.lexsort((-crowd, rank))                # by rank, then most crowded last
    return order[:n], rank, crowd


def run_nsga2(problem, budget, rng, pop_size=40, p_cx=0.9, p_mut=None, sigma=0.2):
    """
    NSGA-II on (V, P, N). Same operators as the GA, but selection uses
    Pareto fronts and crowding distance instead of the scalar fitness.
    """
    tr = Tracker(problem, "NSGA-II")
    p_mut = p_mut or 1.0 / problem.n_genes

    pop = problem.random_population(pop_size, rng)
    obj = problem.evaluate(pop)
    tr.update(pop, obj, problem.fitness(obj))
    Y = objectives_matrix(obj)
    _, rank, crowd = select_nsga(Y, pop_size)

    while tr.used() < budget:
        n_off = min(pop_size, budget - tr.used())
        # tournament on (front, crowding): lower front wins, then larger crowding
        score = rank - crowd / (crowd.max() + 1 if np.isfinite(crowd).all() else 1e9)
        score = np.where(np.isinf(crowd), rank - 0.999, score)
        parents = tournament(score, n_off + n_off % 2, rng)
        children = uniform_crossover(pop[parents[0::2]], pop[parents[1::2]], rng, p_cx)
        children = mutate(children[:n_off], problem, rng, p_mut, sigma)

        obj_c = problem.evaluate(children)
        tr.update(children, obj_c, problem.fitness(obj_c))

        pop_all = np.vstack([pop, children])
        Y_all = np.vstack([Y, objectives_matrix(obj_c)])
        keep, rank_all, crowd_all = select_nsga(Y_all, pop_size)
        pop, Y = pop_all[keep], Y_all[keep]
        rank, crowd = rank_all[keep], crowd_all[keep]
    return tr.result()


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------

def run_sa(problem, budget, rng, T0=10.0, T_end=0.01, sigma=0.2):
    """
    Single-solution SA on the scalar fitness F.
    Starts from 'change nothing'; each step changes one random gene.
    Worse steps are accepted with probability exp(-dF / T); T falls geometrically.
    """
    tr = Tracker(problem, "SA")
    current = np.zeros((1, problem.n_genes))
    obj = problem.evaluate(current)
    F_cur = problem.fitness(obj)[0]
    tr.update(current, obj, np.array([F_cur]))

    alpha = (T_end / T0) ** (1.0 / max(budget - 1, 1))
    T = T0
    span = problem.high - problem.low

    while tr.used() < budget:
        neighbour = current.copy()
        i = rng.integers(problem.n_genes)
        if problem.is_numeric[i]:
            neighbour[0, i] += rng.normal(0.0, sigma * max(span[i], 1e-9))
        else:
            options = np.arange(problem.low[i], problem.high[i] + 1)
            others = options[options != neighbour[0, i]]
            if len(others):
                neighbour[0, i] = rng.choice(others)
        neighbour = problem.repair(neighbour)

        obj_n = problem.evaluate(neighbour)
        F_n = problem.fitness(obj_n)[0]
        tr.update(neighbour, obj_n, np.array([F_n]))

        if F_n <= F_cur or rng.random() < np.exp(-(F_n - F_cur) / T):
            current, F_cur = neighbour, F_n
        T *= alpha
    return tr.result()


METHODS = {
    "Random search": run_random_search,
    "GA": run_ga,
    "NSGA-II": run_nsga2,
    "SA": run_sa,
}


# ---------------------------------------------------------------------------
# Quick comparison on a few applicants
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from problem import Preferences, PerCoFProblem, load_setup

    model, setup = load_setup()
    budget, seeds = 1000, 3

    prefs = Preferences(
        selected=["duration", "amount", "savings", "status"],   # most -> least willing
        directions={"duration": "down", "amount": "down", "savings": "up", "status": "up"},
        bounds={"duration": 12, "amount": 1000},
    )

    print(f"Budget: {budget} evaluations, {seeds} seeds per method\n")
    for idx in range(5):
        x = setup["rejected"].iloc[idx]
        print(f"Applicant {idx}")
        for name, run in METHODS.items():
            Fs, found, times = [], 0, []
            for seed in range(seeds):
                problem = PerCoFProblem(model, setup, x, prefs)
                res = run(problem, budget, np.random.default_rng(seed))
                found += res["found"]
                times.append(res["runtime"])
                if res["found"]:
                    Fs.append(res["F"])
            mean_F = f"{np.mean(Fs):6.2f}" if Fs else "     -"
            print(f"  {name:14s} valid in {found}/{seeds} runs, "
                  f"mean F = {mean_F}, {np.mean(times):.2f} s")
        print()
