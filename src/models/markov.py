"""
markov.py -- Markov chain baseline (model #1, "Ronia's")
=======================================================

The simplest model with actual memory: score "what usually follows the last
`order` cities?" by counting observed transitions in the training trips. No
learning, no gradient -- just counts, which is why it is the first rung above
Popularity on the sophistication ladder
(Popularity -> Markov -> FPMC -> GRU4Rec -> SASRec).

    P_0(i)          = (count(i) + 1) / (total + V)                 unigram
    P_o(i | ctx_o)  = (c(ctx_o, i) + mu * P_{o-1}(i | ctx_{o-1}))
                      / (N(ctx_o) + mu)                            Dirichlet

Smoothing is **Dirichlet-prior interpolation** across orders (unigram -> 1 ->
... -> max_order). A context seen only a few times leans on the lower order, a
frequently seen one trusts its own counts. Uniform / add-k smoothing is
deliberately not used: it adds the same constant to every observed count and so
can never change the ranking.

Benchmark protocol (identical to models_benchmarking.ipynb)
-----------------------------------------------------------
1. Tune with seed 42: fit on TRAIN, score on VALID, select on validation
   NDCG@10. 25 Optuna TPE trials (`TPESampler(seed=42)`), search space
   max_order in [1, 6] and mu in [0.1, 1000] (log-uniform), one mu shared by
   every level.
2. Markov is deterministic and has no epochs, so there is no E_best.
3. Merge train + valid (`merge_train_val`) and refit ONCE (no seeds).
4. Score on test; report the single run.

`tune_markov` does step 1, `final_markov` does steps 3-4.

Ablation: does recursion matter?
--------------------------------
`recursive=False` gives the "highest-order-only" variant:

    recursive (default):  order N -> N-1 -> ... -> 1 -> unigram
    ablation:             order N -> unigram

Only the order-N context is consulted; if it was never seen, the score is the
plain unigram (no help from orders N-1..1). Everything else is identical:
same counts, same Dirichlet formula at the top level, same Optuna protocol
(seed 42, 25 trials, same search space) and the same train+valid refit. The
only difference between the two models is the recursion.

We also track the **fallback rate** (% of trips whose top-order context was
never seen) -- it explains Markov's ceiling in the report.

Every model in the project exposes the same output: a ranked list of city ids
per test trip, scored by the shared `evaluate.py`.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:                                        # works both as a package and flat (Colab)
    from src.evaluate import evaluate
except ImportError:                         # noqa
    from evaluate import evaluate

PAD, UNK, N_RESERVED = 0, 1, 2

# ---- benchmark protocol constants (match the notebook / every other model) ----
SELECT_METRIC = "NDCG@10"     # declared before any search
N_TRIALS = 25                 # same Optuna budget as every other model
SEARCH_SEED = 42              # TPESampler seed
MAX_ORDER = 6                 # search ceiling for the n-gram order
MU_RANGE = (0.1, 1000.0)      # Dirichlet strength, log-uniform
MODEL_NAME = "Markov"                       # results.csv row: recursive model
ABLATION_NAME = "Markov (no recursion)"     # results.csv row: highest-order-only


# ---------------------------------------------------------------------------
# Transition counting
# ---------------------------------------------------------------------------

def _context(cities: Sequence[int], order: int):
    """The last `order` cities as a context key, or None if the trip is too
    short to have one yet."""
    if len(cities) < order:
        return None
    return tuple(cities[-order:])


def precompute_markov_counts(trips, n_cities: int, max_order: int = MAX_ORDER) -> dict:
    """Count every context of length 1..max_order in ONE pass over the trips.

    The tables are reusable: a model of any order <= max_order just slices them,
    so an Optuna search over `order` only ever counts once.
    Returns {"tables": {o: {ctx: Counter}}, "totals": {o: {ctx: int}},
             "counts": unigram count vector, "max_order": int}.
    """
    tables: Dict[int, Dict[tuple, Counter]] = {
        o: defaultdict(Counter) for o in range(1, max_order + 1)}
    counts = np.zeros(n_cities, dtype=np.float64)
    for t in trips:
        cities = t["cities"]
        for i, c in enumerate(cities):
            counts[c] += 1
            for o in range(1, min(max_order, i) + 1):
                tables[o][tuple(cities[i - o:i])][c] += 1
    totals = {o: {ctx: sum(c.values()) for ctx, c in tab.items()}
              for o, tab in tables.items()}
    counts[PAD] = counts[UNK] = 0.0              # never predict PAD/UNK
    return {"tables": tables, "totals": totals, "counts": counts,
            "max_order": max_order}


def build_transition_counts(trips, order: int) -> Dict[tuple, Counter]:
    """context (the `order` cities right before a step) -> Counter(next_city -> count).
    Kept for back-compat; `precompute_markov_counts` is the fast path."""
    table: Dict[tuple, Counter] = defaultdict(Counter)
    for t in trips:
        cities = t["cities"]
        for i in range(order, len(cities)):
            table[tuple(cities[i - order:i])][cities[i]] += 1
    return table


# ---------------------------------------------------------------------------
# Model: recursive Dirichlet-prior interpolation across orders 1..N, bottoming
# out in a unigram prior. mu_o is the strength of the prior inherited from the
# next-lower order.
# ---------------------------------------------------------------------------

class MarkovChain:
    """Order-`order` Markov chain with Dirichlet interpolation to lower orders.

    `mus` is a single float (broadcast to every level -- what the benchmark
    tunes) or a dict {1: mu_1, 2: mu_2, ...}. Every trip gets a full ranking,
    because unseen items inherit the (interpolated) lower-order score.

    `recursive=False` is the ablation: only the order-`order` context is used,
    blended directly with the unigram (orders 1..order-1 are ignored). An
    unseen top-order context therefore falls straight back to the unigram.
    """

    def __init__(self, order: int, n_cities: int, mus=1.0, recursive: bool = True):
        self.order = order
        self.recursive = recursive
        self.n_cities = n_cities
        self.mus = ({o: float(mus) for o in range(1, order + 1)}
                    if not isinstance(mus, dict) else dict(mus))
        self.tables: Dict[int, Dict[tuple, Counter]] = {}
        self.totals: Dict[int, Dict[tuple, int]] = {}
        self.p_uni: np.ndarray = None
        self._fallback_count = 0
        self._total_count = 0

    def fit(self, trips, precomputed: dict | None = None) -> "MarkovChain":
        """Fit on `trips`, or reuse `precomputed` counts (see
        `precompute_markov_counts`) so a search over order/mu never recounts."""
        pre = precomputed or precompute_markov_counts(trips, self.n_cities, self.order)
        if pre["max_order"] < self.order:
            raise ValueError(f"precomputed counts only go to order {pre['max_order']}")
        levels = self._levels()
        self.tables = {o: pre["tables"][o] for o in levels}
        self.totals = {o: pre["totals"][o] for o in levels}
        p = pre["counts"] + 1.0                   # add-one on the unigram only
        p[PAD] = p[UNK] = 0.0
        self.p_uni = p / p.sum()
        return self

    def _levels(self) -> List[int]:
        """Orders actually used: 1..order (recursive) or just [order] (ablation)."""
        return list(range(1, self.order + 1)) if self.recursive else [self.order]

    def _scores_for_trip(self, cities: Sequence[int]) -> Tuple[np.ndarray, bool]:
        """Full score vector over the catalog for one trip + fallback flag."""
        p = self.p_uni
        top_hit = False
        for o in self._levels():
            ctx = _context(cities, o)
            if ctx is None or ctx not in self.tables[o]:
                continue                          # level unavailable -> inherit p
            if o == self.order:
                top_hit = True
            counter, n = self.tables[o][ctx], self.totals[o][ctx]
            mu = self.mus[o]
            q = p * (mu / (n + mu))
            idx = np.fromiter(counter.keys(), dtype=np.int64, count=len(counter))
            val = np.fromiter(counter.values(), dtype=np.float64, count=len(counter))
            q[idx] += val / (n + mu)
            p = q
        p = p.copy() if p is self.p_uni else p
        p[PAD] = p[UNK] = -np.inf
        return p, (not top_hit)

    def predict_ranked(self, trips, k: int = 100) -> List[list]:
        ranked_lists = []
        self._fallback_count = 0
        self._total_count = 0
        for t in trips:
            scores, fell_back = self._scores_for_trip(t["cities"])
            part = np.argpartition(-scores, kth=k - 1)[:k]
            ranked_lists.append(part[np.argsort(-scores[part])].tolist())
            self._fallback_count += int(fell_back)
            self._total_count += 1
        return ranked_lists

    @property
    def fallback_rate(self) -> float:
        """Fraction of the last `predict_ranked` call's trips whose top-order
        context was never seen in training (served by a lower order/unigram)."""
        return self._fallback_count / self._total_count if self._total_count else 0.0

    @property
    def n_params(self) -> int:
        """Distinct (context, next-city) counts stored."""
        return sum(len(counter) for table in self.tables.values()
                   for counter in table.values())


# ---------------------------------------------------------------------------
# Train + score on validation only (used by the search, never touches test)
# ---------------------------------------------------------------------------

def train_markov(data, order: int = 1, mus=1.0, k: int = 100,
                 verbose: bool = True, select_metric: str = SELECT_METRIC,
                 precomputed: dict | None = None, recursive: bool = True):
    """Fit on data['train'], score on data['valid']. Returns (model, val_score)."""
    model = MarkovChain(order=order, n_cities=data["n_cities"],
                        mus=mus, recursive=recursive).fit(data["train"], precomputed=precomputed)
    val_targets = [t["target_city"] for t in data["valid"]]
    val_score = 0.0
    if val_targets:
        val_ranked = model.predict_ranked(data["valid"], k=k)
        val_score = evaluate(val_ranked, val_targets)[select_metric]
    if verbose:
        print(f"  order={order} recursive={recursive} mus={model.mus} val_{select_metric}={val_score:.4f}")
    return model, val_score


# ---------------------------------------------------------------------------
# Step 1 of the protocol: Optuna search (TPE seed 42, 25 trials, validation only)
# ---------------------------------------------------------------------------

def tune_markov(data, n_trials: int = N_TRIALS, seed: int = SEARCH_SEED,
                max_order: int = MAX_ORDER, k: int = 100,
                select_metric: str = SELECT_METRIC, show_progress_bar: bool = True,
                recursive: bool = True):
    """Optuna over (max_order, mu) on validation `select_metric`.

    `recursive=False` runs the identical search for the highest-order-only
    ablation (same sampler seed, budget and search space).

    Same budget and sampler as every other model. Counts are precomputed once up
    to `max_order`, so each trial is just a re-weighting. Returns the finished
    `optuna.Study`; `study.best_params` = {"max_order": ..., "mu": ...}.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pre = precompute_markov_counts(data["train"], data["n_cities"], max_order)

    def objective(trial):
        order = trial.suggest_int("max_order", 1, max_order)
        mu = trial.suggest_float("mu", *MU_RANGE, log=True)
        _, v = train_markov(data, order=order, mus=mu, k=k, verbose=False,
                            select_metric=select_metric, precomputed=pre,
                            recursive=recursive)
        return v

    name = "Markov (Dirichlet)" if recursive else "Markov (Dirichlet, no recursion)"
    study = optuna.create_study(study_name=name, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    print(f"{name}: best val {select_metric} = {study.best_value:.4f} "
          f"({len(study.trials)} trials, {time.time() - t0:.0f}s)")
    print(f"  params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Steps 3-4: refit on train+valid once (deterministic), score on test
# ---------------------------------------------------------------------------

def final_markov(data, params: dict, k: int = 100, recursive: bool = True):
    """Refit on TRAIN + VALID with the tuned config and score on test.

    `params` = study.best_params ({"max_order", "mu"}). Deterministic, so there
    is exactly one run (no seeds, no epochs). Returns (metrics, ranked_lists).
    """
    try:                                    # same import style as the evaluate import
        from src.preprocessing import merge_train_val
    except ImportError:                     # noqa
        from preprocessing import merge_train_val
    t0 = time.time()
    merged = merge_train_val(data)
    model = MarkovChain(order=params["max_order"], n_cities=data["n_cities"],
                        mus=params["mu"], recursive=recursive).fit(merged["train"])
    ranked = model.predict_ranked(data["test"], k=k)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = model.n_params
    metrics["fallback_rate"] = round(model.fallback_rate, 4)
    return metrics, ranked


def run_markov(data, order: int = 1, mus=1.0, k: int = 100, verbose: bool = True,
               select_metric: str = SELECT_METRIC, recursive: bool = True):
    """Quick single run: fit on data['train'], predict test, score. Returns
    (metrics, ranked_lists, val_score), the same shape as `run_gru4rec`.

    NOTE: fit on data['train'] only. The benchmark's final numbers come from
    `final_markov` (train + valid), not from here.
    """
    t0 = time.time()
    model, val_score = train_markov(data, order=order, mus=mus, k=k,
                                    verbose=verbose, select_metric=select_metric,
                                    recursive=recursive)
    ranked = model.predict_ranked(data["test"], k=k)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = model.n_params
    metrics["fallback_rate"] = round(model.fallback_rate, 4)
    return metrics, ranked, val_score


def select_markov(data, orders: Sequence[int] = tuple(range(1, MAX_ORDER + 1)),
                  mu_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
                  k: int = 100, verbose: bool = True):
    """Back-compat grid search (orders 1..6 x mu grid by validation NDCG@10).
    The protocol uses `tune_markov`. Returns (metrics, ranked, val_ndcg, config)."""
    pre = precompute_markov_counts(data["train"], data["n_cities"], max(orders))
    best = None
    for order in orders:
        for mu in mu_grid:
            _, v = train_markov(data, order=order, mus=mu, k=k, verbose=verbose,
                                precomputed=pre)
            if best is None or v > best[0]:
                best = (v, order, mu)
    _, order, mu = best
    metrics, ranked, val = run_markov(data, order=order, mus=mu, k=k, verbose=False)
    return metrics, ranked, val, {"order": order, "mu": mu}
