"""
evaluate.py
===========
Shared scorer for the Booking.com MDT project. EVERY model is scored with this
one module so the comparison is apples-to-apples.

The contract (what a model must produce)
----------------------------------------
For the test trips -- in the SAME order as `data['test']` from preprocessing --
each model produces a *ranked list of city indices*, best guess first:

    ranked_lists[i] = [best_city_idx, 2nd_best, 3rd_best, ...]   # len >= max(k)

Then:

    from src.evaluate import evaluate
    targets = [t['target_city'] for t in data['test']]
    metrics = evaluate(ranked_lists, targets)

That is the only interface. How the ranking was produced (frequency counts,
Markov transitions, learned embeddings, attention) does not matter -- the scorer
only looks at where the true city lands in the list.

Metrics
-------
HR@k, NDCG@k, MRR for k in {1, 5, 10}, plus Accuracy@4 (= HR@4) as a sanity peg:
the built-in Popularity baseline should reproduce the repo's demo (~0.053), which
proves the scoring is correct.

Ranking is against the full city catalog. City index 0 is PAD/UNK and is never a
valid prediction; if a trip's true city is 0 (unseen in training) it is an
automatic miss, exactly as it would be in deployment.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Sequence

import numpy as np

DEFAULT_KS = (1, 5, 10)
PAD = 0            # padding index -- never a valid prediction
UNK = 1            # unknown-city index -- never a valid prediction
RESERVED = (PAD, UNK)


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def evaluate(ranked_lists: Sequence[Sequence[int]],
             targets: Sequence[int],
             ks: Sequence[int] = DEFAULT_KS,
             also_acc4: bool = True) -> Dict[str, float]:
    """Compute HR@k, NDCG@k, MRR (and Accuracy@4) over all test trips.

    Parameters
    ----------
    ranked_lists : per trip, city indices ordered best-first (length >= max(ks)).
    targets      : per trip, the true city index (0 if unseen -> always a miss).
    ks           : cut-offs for HR / NDCG.
    also_acc4    : also return HR@4 under the name 'Accuracy@4'.

    Returns a flat dict, e.g. {'HR@1':.., 'HR@5':.., 'NDCG@10':.., 'MRR':..}.
    """
    if len(ranked_lists) != len(targets):
        raise ValueError(
            f"{len(ranked_lists)} ranked lists but {len(targets)} targets"
        )

    ks = tuple(sorted(set(ks) | ({4} if also_acc4 else set())))
    kmax = max(ks)
    hr = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = 0.0
    n = len(targets)

    for ranked, tgt in zip(ranked_lists, targets):
        # position (1-based) of the true city within the top-kmax of the list
        pos = None
        if tgt not in RESERVED:                 # PAD/UNK targets are always misses
            for i, c in enumerate(ranked[:kmax]):
                if c == tgt:
                    pos = i + 1
                    break
        if pos is None:
            continue
        mrr += 1.0 / pos
        gain = 1.0 / math.log2(pos + 1)  # single relevant item -> IDCG = 1
        for k in ks:
            if pos <= k:
                hr[k] += 1.0
                ndcg[k] += gain

    out: Dict[str, float] = {}
    for k in ks:
        if k == 4 and also_acc4:
            out["Accuracy@4"] = hr[4] / n
            continue
        if k in DEFAULT_KS:
            out[f"HR@{k}"] = hr[k] / n
            out[f"NDCG@{k}"] = ndcg[k] / n
    out["MRR"] = mrr / n
    out["n_trips"] = n
    return out


# ---------------------------------------------------------------------------
# Helper: turn a full score vector/matrix into ranked lists
# ---------------------------------------------------------------------------

def topk_from_scores(scores: np.ndarray, k: int = 100,
                     exclude: Sequence[int] = RESERVED) -> List[np.ndarray]:
    """Convert a [n_trips, n_cities] score matrix into top-k ranked index lists.

    Used by the embedding models (FPMC / SASRec) that produce a score per city.
    `exclude` masks indices that must never be predicted (PAD/UNK).
    """
    scores = np.asarray(scores, dtype=float).copy()
    for idx in exclude:
        scores[:, idx] = -np.inf
    k = min(k, scores.shape[1])
    # argpartition for speed, then sort the top-k slice
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    ranked = []
    for row, cols in zip(scores, part):
        ranked.append(cols[np.argsort(-row[cols])])
    return ranked


# ---------------------------------------------------------------------------
# Sliced evaluation (metrics per subgroup -- e.g. by trip length or popularity)
# ---------------------------------------------------------------------------

def evaluate_sliced(ranked_lists, targets, group_ids, ks=DEFAULT_KS):
    """Return {group_label: metrics} by bucketing trips on `group_ids`.

    Lets the report answer questions like "does SASRec win on long trips?" or
    "does it beat Markov on rare (tail) destinations?".
    """
    from collections import defaultdict
    buckets = defaultdict(lambda: ([], []))
    for r, t, g in zip(ranked_lists, targets, group_ids):
        buckets[g][0].append(r)
        buckets[g][1].append(t)
    return {g: evaluate(rl, tg, ks=ks) for g, (rl, tg) in
            sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def city_frequency(data) -> np.ndarray:
    """Count how often each city appears across the TRAIN trips."""
    counts = np.zeros(data["n_cities"])
    for trip in data["train"]:
        for c in trip["cities"]:
            counts[c] += 1
    return counts


# ---------------------------------------------------------------------------
# Built-in Popularity baseline (the 0th model + correctness peg)
# ---------------------------------------------------------------------------

def popularity_topk(data: dict, k: int = 100) -> np.ndarray:
    """Rank cities by their frequency in the TRAIN trips (same list for everyone)."""
    n_cities = data["n_cities"]
    counts = np.zeros(n_cities, dtype=np.int64)
    for trip in data["train"]:
        for c in trip["cities"]:
            counts[c] += 1
    counts[PAD] = counts[UNK] = -1  # never predict PAD/UNK
    k = min(k, n_cities)
    top = np.argpartition(-counts, kth=k - 1)[:k]   # top-k indices (unordered)
    return top[np.argsort(-counts[top])]            # ordered best-first


def evaluate_popularity(data: dict, k: int = 100,
                        ks: Sequence[int] = DEFAULT_KS) -> Dict[str, float]:
    """Score the Popularity baseline. Its Accuracy@4 should reproduce the
    repo demo (~0.053) -- if it does, the scorer is trustworthy."""
    top = list(popularity_topk(data, k=k))
    targets = [t["target_city"] for t in data["test"]]
    ranked_lists = [top] * len(targets)  # same ranking for every trip
    return evaluate(ranked_lists, targets, ks=ks)


# ---------------------------------------------------------------------------
# Results logging (shared results.csv all models append to)
# ---------------------------------------------------------------------------

RESULT_COLUMNS = ["model", "HR@1", "HR@5", "HR@10",
                  "NDCG@5", "NDCG@10", "MRR", "Accuracy@4", "n_trips"]


def append_result(path: str, model: str, metrics: Dict[str, float]) -> None:
    """Append one model's metrics as a row to a shared results.csv."""
    row = {"model": model}
    row.update({c: metrics.get(c, "") for c in RESULT_COLUMNS if c != "model"})
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Reading results.csv back (mean rows + "<model> (std)" rows)
# ---------------------------------------------------------------------------

STD_SUFFIX = " (std)"


def load_results(path: str):
    """Read results.csv and split it into (means, stds) DataFrames indexed by model.

    The benchmark protocol logs, per stochastic model, a mean row ("SASRec-no-side")
    and a spread row ("SASRec-no-side (std)", 5-seed sample std). Deterministic
    models (Markov) and any model whose std row has not been added yet simply have
    no std row: they get NaN in `stds` rather than raising.

    If a row was logged more than once, the LAST one wins (re-runs overwrite).
    """
    import pandas as pd
    df = pd.read_csv(path).drop_duplicates("model", keep="last").set_index("model")
    is_std = df.index.str.endswith(STD_SUFFIX)
    means = df[~is_std]
    stds = df[is_std].copy()
    stds.index = stds.index.str[: -len(STD_SUFFIX)]
    stds = stds.reindex(means.index)           # NaN where no std row exists
    return means, stds


# ---------------------------------------------------------------------------
# Fast bootstrap over test trips (evaluation uncertainty)
# ---------------------------------------------------------------------------

def target_ranks(ranked_lists, targets, kmax: int = 10) -> np.ndarray:
    """1-based rank of the true city within the top-`kmax` of each list (0 = miss).
    Same PAD/UNK rule as `evaluate`. Computed once, so bootstrapping is cheap."""
    ranks = np.zeros(len(targets), dtype=np.int32)
    for i, (ranked, tgt) in enumerate(zip(ranked_lists, targets)):
        if tgt in RESERVED:
            continue
        for j, c in enumerate(ranked[:kmax]):
            if c == tgt:
                ranks[i] = j + 1
                break
    return ranks


def metric_from_ranks(ranks: np.ndarray, metric: str = "NDCG@10") -> float:
    """HR@k / NDCG@k / MRR from `target_ranks` output (matches `evaluate`)."""
    hit = ranks > 0
    if metric == "MRR":
        return float(np.where(hit, 1.0 / np.maximum(ranks, 1), 0.0).mean())
    kind, k = metric.split("@")
    k = int(k)
    inside = hit & (ranks <= k)
    if kind == "HR":
        return float(inside.mean())
    if kind == "NDCG":
        return float(np.where(inside, 1.0 / np.log2(np.maximum(ranks, 1) + 1.0), 0.0).mean())
    raise ValueError(f"unsupported metric {metric!r}")


def bootstrap_ci(ranked_lists, targets, metric: str = "NDCG@10",
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over TEST TRIPS for one model.
    Returns (mean, (lo, hi)). Vectorised: ~1000 resamples take a second or two."""
    ranks = target_ranks(ranked_lists, targets)
    rng = np.random.default_rng(seed)
    n = len(ranks)
    vals = np.array([metric_from_ranks(ranks[rng.integers(0, n, n)], metric)
                     for _ in range(n_boot)])
    return float(vals.mean()), tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def paired_bootstrap_diff(ranked_a, ranked_b, targets, metric: str = "NDCG@10",
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap of metric(A) - metric(B): the SAME resampled trips score
    both models, which is much tighter than comparing two separate CIs.
    Returns (observed_diff, (lo, hi)); a CI excluding 0 = the gap is not test-set noise."""
    ra, rb = target_ranks(ranked_a, targets), target_ranks(ranked_b, targets)
    rng = np.random.default_rng(seed)
    n = len(ra)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_from_ranks(ra[idx], metric) - metric_from_ranks(rb[idx], metric)
    obs = metric_from_ranks(ra, metric) - metric_from_ranks(rb, metric)
    return obs, tuple(np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# ---------------------------------------------------------------------------
# Smoke test:  python -m src.evaluate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # true cities and a hand-made ranking to check the math
    ranked = [
        [5, 3, 9, 1, 2],   # true=3 -> pos 2
        [7, 8, 2, 4, 6],   # true=4 -> pos 4
        [1, 2, 3, 4, 5],   # true=9 -> not in top -> miss
    ]
    targets = [3, 4, 9]
    m = evaluate(ranked, targets, ks=(1, 5, 10))
    print("metrics:", {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in m.items()})

    # checks
    assert abs(m["HR@1"] - 0.0) < 1e-9              # neither hit at rank 1
    assert abs(m["HR@5"] - 2 / 3) < 1e-9            # two of three within top 5
    assert abs(m["MRR"] - (1 / 2 + 1 / 4) / 3) < 1e-9
    expected_ndcg5 = (1 / math.log2(3) + 1 / math.log2(5)) / 3
    assert abs(m["NDCG@5"] - expected_ndcg5) < 1e-9

    # topk_from_scores sanity
    scores = np.array([[0.1, 0.9, 0.5, 0.2],
                       [0.3, 0.0, 0.8, 0.4]])
    tk = topk_from_scores(scores, k=2, exclude=(0,))
    assert list(tk[0]) == [1, 2] and list(tk[1]) == [2, 3]

    # fast-bootstrap helpers must agree with evaluate()
    rk = target_ranks(ranked, targets)
    for met in ("HR@1", "HR@5", "NDCG@5", "MRR"):
        assert abs(metric_from_ranks(rk, met) - m[met]) < 1e-9, met
    mean_, (lo_, hi_) = bootstrap_ci(ranked, targets, "HR@5", n_boot=50)
    assert lo_ <= mean_ <= hi_
    d_, (dlo, dhi) = paired_bootstrap_diff(ranked, ranked, targets, n_boot=20)
    assert d_ == 0 and dlo == 0 and dhi == 0
    print("ALL SMOKE-TEST ASSERTIONS PASSED")
