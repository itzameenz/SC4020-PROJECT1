"""
markov.py -- Markov chain baselines (model #1, "Ronia's")
===========================================================

The simplest model with actual memory: score "what usually follows city `l`
(order-1) or the pair `(l2, l1)` (order-2)?" by counting observed transitions
in the training trips. No learning, no gradient -- just counts, which is why
it is the first rung above Popularity on the sophistication ladder
(Popularity -> Markov -> FPMC -> GRU4Rec -> SASRec).

    score(i | context) = count(context -> i) + smoothing     # add-k / Laplace

Two project-specific pieces (both feed the report):

* Order-2 with fallback -- a context the model never saw at order-2 falls back
  to order-1, and an order-1 context it never saw falls back to global
  Popularity. Every trip always gets a full ranked list this way. We track the
  **fallback rate** (% of test trips that could not be served by the model's
  own top order) -- this is what explains Markov's ceiling in the report.

* Add-k (Laplace) smoothing -- a single constant `smoothing` added to every
  observed count, so a context seen once isn't treated as absolutely certain.

Tuning protocol (matches every other model): order in {1, 2} and the smoothing
constant are chosen by **validation NDCG@10** via `select_markov`, exactly
like the neural models' hyperparameter grids -- the test set is never touched
during selection.

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


# ---------------------------------------------------------------------------
# Transition counting
# ---------------------------------------------------------------------------

def _context(cities: Sequence[int], order: int):
    """The last `order` cities as a context key, or None if the trip is too
    short to have one yet."""
    if len(cities) < order:
        return None
    return tuple(cities[-order:])


def build_transition_counts(trips, order: int) -> Dict[tuple, Counter]:
    """context (the `order` cities right before a step) -> Counter(next_city -> count).

    Built once over every consecutive window in every training trip, so an
    order-2 table also implicitly needs order-1 data collected separately
    (see `MarkovChain.fit`, which builds both).
    """
    table: Dict[tuple, Counter] = defaultdict(Counter)
    for t in trips:
        cities = t["cities"]
        for i in range(order, len(cities)):
            ctx = tuple(cities[i - order:i])
            table[ctx][cities[i]] += 1
    return table


# ---------------------------------------------------------------------------
# Model: an order-k table with add-k smoothing and a fallback chain
# ---------------------------------------------------------------------------

class MarkovChain:
    """Order-`order` transition table, order-(order-1) ... order-1 tables kept
    alongside as a fallback chain, and a global Popularity ranking as the last
    resort. This guarantees every trip gets a full-length ranked list even
    when its exact context was never observed in training.
    """

    def __init__(self, order: int, n_cities: int, smoothing: float = 1.0):
        self.order = order
        self.n_cities = n_cities
        self.smoothing = smoothing
        self.tables: Dict[int, Dict[tuple, Counter]] = {}
        self.pop_rank: np.ndarray = None
        self._fallback_count = 0
        self._total_count = 0

    def fit(self, trips) -> "MarkovChain":
        for o in range(1, self.order + 1):
            self.tables[o] = build_transition_counts(trips, o)

        counts = np.zeros(self.n_cities, dtype=np.int64)
        for t in trips:
            for c in t["cities"]:
                counts[c] += 1
        counts[PAD] = counts[UNK] = -1          # never predict PAD/UNK
        self.pop_rank = np.argsort(-counts)
        return self

    def _popularity_tail(self):
        return (c for c in self.pop_rank if c not in (PAD, UNK))

    def _rank_for_trip(self, cities: Sequence[int], k: int) -> Tuple[list, bool]:
        """Rank candidates for one trip; also report whether the model had to
        fall back below its own configured order (or all the way to Popularity)."""
        for o in range(self.order, 0, -1):
            ctx = _context(cities, o)
            if ctx is not None and ctx in self.tables[o]:
                counter = self.tables[o][ctx]
                scored = {c: n + self.smoothing for c, n in counter.items()
                          if c not in (PAD, UNK)}
                ranked = sorted(scored, key=scored.get, reverse=True)
                if len(ranked) < k:                     # pad out with popularity
                    seen = set(ranked)
                    for c in self._popularity_tail():
                        if c not in seen:
                            ranked.append(c)
                            seen.add(c)
                        if len(ranked) >= k:
                            break
                return ranked[:k], (o != self.order)
        # no order matched at all -> pure Popularity fallback
        return list(self._popularity_tail())[:k], True

    def predict_ranked(self, trips, k: int = 100) -> List[list]:
        ranked_lists = []
        self._fallback_count = 0
        self._total_count = 0
        for t in trips:
            ranked, fell_back = self._rank_for_trip(t["cities"], k)
            ranked_lists.append(ranked)
            self._fallback_count += int(fell_back)
            self._total_count += 1
        return ranked_lists

    @property
    def fallback_rate(self) -> float:
        """Fraction of the last `predict_ranked` call's trips that could not be
        served by the model's own top-order context (needed a lower order, or
        Popularity). Explains why Markov plateaus: it has no answer beyond
        what it has literally seen before."""
        return self._fallback_count / self._total_count if self._total_count else 0.0

    @property
    def n_params(self) -> int:
        """Number of distinct (context, next-city) counts stored -- the
        count-based analogue of a neural model's parameter count, for the
        accuracy-vs-cost comparison in the report."""
        return sum(len(counter) for table in self.tables.values()
                   for counter in table.values())


# ---------------------------------------------------------------------------
# Train + score on validation only (used by the grid search, never touches test)
# ---------------------------------------------------------------------------

def train_markov(data, order: int = 1, smoothing: float = 1.0, k: int = 100,
                 verbose: bool = True):
    """Fit on train, score on validation. Returns (model, val_ndcg)."""
    model = MarkovChain(order=order, n_cities=data["n_cities"],
                        smoothing=smoothing).fit(data["train"])
    val_targets = [t["target_city"] for t in data["valid"]]
    val_ndcg = 0.0
    if val_targets:
        val_ranked = model.predict_ranked(data["valid"], k=k)
        val_ndcg = evaluate(val_ranked, val_targets)["NDCG@10"]
    if verbose:
        print(f"  order={order} smoothing={smoothing:<5} val_NDCG@10={val_ndcg:.4f}")
    return model, val_ndcg


def run_markov(data, order: int = 1, smoothing: float = 1.0, k: int = 100,
               verbose: bool = True):
    """Train, predict on test, and score. Returns (metrics, ranked_lists, val_ndcg),
    the same shape as `run_fpmc` / `run_sasrec` / `run_gru4rec`."""
    t0 = time.time()
    model, val_ndcg = train_markov(data, order=order, smoothing=smoothing,
                                   k=k, verbose=verbose)
    ranked = model.predict_ranked(data["test"], k=k)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = model.n_params
    metrics["fallback_rate"] = round(model.fallback_rate, 4)
    return metrics, ranked, val_ndcg


# ---------------------------------------------------------------------------
# Grid search: order x smoothing, selected by validation NDCG@10
# ---------------------------------------------------------------------------

def select_markov(data, orders: Sequence[int] = (1, 2),
                  smoothings: Sequence[float] = (0.1, 1.0, 5.0),
                  k: int = 100, verbose: bool = True):
    """Grid search over order and add-k smoothing (Markov's only real
    hyperparameters), same protocol as every other model: picked by
    validation NDCG@10, test touched only once at the end.

    Returns (metrics, ranked_lists, val_ndcg, best_config).
    """
    best = None  # (val_ndcg, order, smoothing)
    for order in orders:
        for s in smoothings:
            _, val_ndcg = train_markov(data, order=order, smoothing=s, k=k,
                                       verbose=verbose)
            if best is None or val_ndcg > best[0]:
                best = (val_ndcg, order, s)

    _, order, smoothing = best
    if verbose:
        print(f"selected: order={order} smoothing={smoothing} "
              f"(val_NDCG@10={best[0]:.4f})")
    metrics, ranked, val_ndcg = run_markov(data, order=order, smoothing=smoothing,
                                           k=k, verbose=False)
    return metrics, ranked, val_ndcg, {"order": order, "smoothing": smoothing}
