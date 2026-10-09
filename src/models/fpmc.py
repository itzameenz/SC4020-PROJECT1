"""
fpmc.py -- Factorized Personalized Markov Chains (model #2, "ours")
==================================================================

FPMC fuses two ideas into one score for "given this traveller, sitting in city
`l`, how likely is city `i` next?":

    score(u, l, i) = < user_factor(u) , item_factor_U(i) >     # personalization
                   + < last_factor(l) , item_factor_L(i) >     # Markov transition

The first term is matrix factorization (who tends to like what); the second is a
*factorized* Markov chain (what usually follows what).

Benchmark protocol (identical to gru4rec.py -- constants are imported from there)
--------------------------------------------------------------------------------
* Loss: full-catalogue **cross-entropy** over every candidate next city
  (`loss="ce"`), same as GRU4Rec / SASRec. The classic FPMC objective, BPR
  against one sampled negative, is still available as `loss="bpr"`.
* Optimiser: **AdamW** (`build_adamw`). lr and weight_decay are tuned.
* Early stopping on validation NDCG@10: `MAX_EPOCHS=20`, `PATIENCE=3`.
* Search (`tune_fpmc`): 25 Optuna TPE trials, `TPESampler(seed=42)`,
  `MedianPruner(n_warmup_steps=2)`, validation only. Only the knobs of the
  unified space that exist in FPMC are searched:
      embed_dim {64,128,256} | lr [1e-4,1e-2] log | weight_decay [1e-6,1e-3] log
      batch_size {64,128,256}
  FPMC has no layers, no dropout and reads only the last city, so
  num_layers / dropout / max_len do not apply.
* Final runs (`final_fpmc`): record E_best of the winning trial, merge
  train + valid, train 5 fresh models (seeds 1, 7, 13, 2024, 99) for exactly
  E_best epochs, score on test, report mean +/- std.

Two project-specific additions (both documented in the report):

* Personalization axis -- `axis="user"` uses the raw `user_id`; `axis="persona"`
  uses (booker_country, device_class). Because almost every user_id occurs once,
  the raw variant cannot generalise to test; the persona recurs across trips.
  Running both is our headline ablation.

* Position feature -- FPMC has no built-in notion of where in a trip a step sits
  (SASRec gets this for free). We add a position-bucket embedding to the user side:
      user_side = user_factor(u) + position_factor(bucket)

Every model in the project exposes the same output: a ranked list of city ids per
test trip, scored by the shared `evaluate.py`.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                        # works both as a package and flat (Colab)
    from src.evaluate import evaluate
    from src.models.gru4rec import (set_seed, build_adamw, SELECT_METRIC, N_TRIALS,
                                    MAX_EPOCHS, PATIENCE, SEARCH_SEED, FINAL_SEEDS)
except ImportError:                         # noqa
    from evaluate import evaluate
    from gru4rec import (set_seed, build_adamw, SELECT_METRIC, N_TRIALS,
                         MAX_EPOCHS, PATIENCE, SEARCH_SEED, FINAL_SEEDS)

PAD, UNK, N_RESERVED = 0, 1, 2


# ---------------------------------------------------------------------------
# Turn trips into (user, last_city, next_position_bucket, next_city) examples
# ---------------------------------------------------------------------------

def build_transitions(trips, axis: str):
    """Every consecutive city pair in every trip becomes one training example."""
    u, prev, posb, nxt = [], [], [], []
    for t in trips:
        cities, buckets, uid = t["cities"], t["pos_bucket"], t[axis]
        for i in range(1, len(cities)):
            u.append(uid); prev.append(cities[i - 1])
            posb.append(buckets[i]); nxt.append(cities[i])
    return (np.asarray(u), np.asarray(prev), np.asarray(posb), np.asarray(nxt))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FPMC(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_buckets: int, dim: int = 64):
        super().__init__()
        self.UI = nn.Embedding(n_users, dim)     # user  -> item (personalization)
        self.IU = nn.Embedding(n_items, dim)     # item  -> user side
        self.LI = nn.Embedding(n_items, dim)     # last  -> transition (from side)
        self.IL = nn.Embedding(n_items, dim)     # item  -> transition (to side)
        self.POS = nn.Embedding(n_buckets, dim)  # position bucket -> user side
        for emb in (self.UI, self.IU, self.LI, self.IL, self.POS):
            nn.init.normal_(emb.weight, std=0.01)

    def _sides(self, u, l, pos):
        """Return the user-side vector (u + position) and the last-side vector."""
        return self.UI(u) + self.POS(pos), self.LI(l)

    def score_pairs(self, u, l, pos, i):
        """Score specific candidate items i (same shape as u/l)."""
        cu, cl = self._sides(u, l, pos)
        return (cu * self.IU(i)).sum(-1) + (cl * self.IL(i)).sum(-1)

    def score_all(self, u, l, pos):
        """Score every item in the catalog -> [batch, n_items]."""
        cu, cl = self._sides(u, l, pos)
        return cu @ self.IU.weight.t() + cl @ self.IL.weight.t()


# ---------------------------------------------------------------------------
# Prediction -> ranked lists (chunked so we never build a huge score matrix)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_ranked(model, trips, axis, k=100, batch=1024, device="cpu"):
    model.eval()
    ranked: List[list] = []
    for s in range(0, len(trips), batch):
        chunk = [t for t in trips[s:s + batch] if len(t["cities"]) > 0]
        u = torch.tensor([t[axis] for t in chunk], device=device)
        l = torch.tensor([t["cities"][-1] for t in chunk], device=device)
        pos = torch.tensor([t["pos_bucket"][-1] for t in chunk], device=device)
        scores = model.score_all(u, l, pos)
        scores[:, PAD] = -np.inf
        scores[:, UNK] = -np.inf
        top = torch.topk(scores, k, dim=1).indices.cpu().tolist()
        ranked.extend(top)
    return ranked


# ---------------------------------------------------------------------------
# Training with early stopping on validation NDCG@10
# ---------------------------------------------------------------------------

def train_fpmc(data, axis="persona", dim=64, loss="ce", epochs=MAX_EPOCHS, lr=1e-3,
               batch=256, weight_decay=1e-6, seed=SEARCH_SEED, patience=PATIENCE,
               fixed_epochs=None, select_metric=SELECT_METRIC, trial=None,
               device="cpu", verbose=True):
    """Train one FPMC. Returns (model, best validation score, best epoch).

    `fixed_epochs` => final-fit mode: train exactly that many epochs on
    data['train'] (pass the train+valid merge), no validation, no early stopping,
    keep the LAST-epoch weights; the returned score is NaN.
    `trial` => an optuna.Trial: report val score each epoch and allow pruning.
    """
    set_seed(seed)
    n_items = data["n_cities"]
    if axis == "user":
        n_users = len(data["encoders"]["user_id"]) + N_RESERVED
    else:
        n_users = len(data["persona_encoder"]) + N_RESERVED
    n_buckets = 5

    u, prev, posb, nxt = build_transitions(data["train"], axis)
    u = torch.tensor(u, device=device); prev = torch.tensor(prev, device=device)
    posb = torch.tensor(posb, device=device); nxt = torch.tensor(nxt, device=device)
    n = len(u)

    model = FPMC(n_users, n_items, n_buckets, dim).to(device)
    opt = build_adamw(model, lr, weight_decay)

    val_targets = [t["target_city"] for t in data["valid"]]
    best_score, best_state, stale, best_epoch = -1.0, None, 0, 0
    history = []
    if fixed_epochs is not None:
        epochs = fixed_epochs

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            if loss == "ce":
                logits = model.score_all(u[idx], prev[idx], posb[idx])   # [B, n_items]
                l = F.cross_entropy(logits, nxt[idx])
            else:                                                       # "bpr": 1 sampled negative
                neg = torch.randint(N_RESERVED, n_items, (len(idx),), device=device)
                pos_s = model.score_pairs(u[idx], prev[idx], posb[idx], nxt[idx])
                neg_s = model.score_pairs(u[idx], prev[idx], posb[idx], neg)
                l = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item() * len(idx)

        if fixed_epochs is not None:
            if verbose:
                print(f"  epoch {ep:2d}  loss={tot / n:.4f}  (fixed-epoch final fit)")
            continue

        score = 0.0
        if val_targets:
            ranked = predict_ranked(model, data["valid"], axis, device=device)
            score = evaluate(ranked, val_targets)[select_metric]
        history.append(score)
        if verbose:
            print(f"  epoch {ep:2d}  loss={tot / n:.4f}  val_{select_metric}={score:.4f}")

        if trial is not None:
            trial.report(score, ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if score > best_score:
            best_score, best_epoch, stale = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    if fixed_epochs is not None:
        best_score, best_epoch = float("nan"), fixed_epochs   # no validation in final fit
    model.best_epoch = best_epoch
    model.history = history
    return model, best_score, best_epoch


# ---------------------------------------------------------------------------
# Step 1 of the protocol: Optuna search (TPE seed 42, 25 trials, validation only)
# ---------------------------------------------------------------------------

def suggest_params(trial) -> dict:
    """The part of the unified search space that exists in FPMC (no layers,
    no dropout, no sequence length -- it only reads the last city)."""
    return {
        "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "embed_dim":    trial.suggest_categorical("embed_dim", [64, 128, 256]),
        "batch_size":   trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def _train_kwargs(p: dict) -> dict:
    """Optuna param names -> train_fpmc arguments."""
    return dict(dim=p["embed_dim"], lr=p["lr"], batch=p["batch_size"],
                weight_decay=p["weight_decay"])


def tune_fpmc(data, axis="persona", n_trials: int = N_TRIALS, seed: int = SEARCH_SEED,
              loss: str = "ce", device: str = "cpu", show_progress_bar: bool = True):
    """Optuna search over the unified space on validation NDCG@10. Returns the
    finished `optuna.Study`; `study.best_params` is the winning config and
    `study.best_trial.user_attrs["best_epoch"]` is E_best."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = suggest_params(trial)
        _, score, epoch = train_fpmc(data, axis=axis, loss=loss, trial=trial,
                                     device=device, verbose=False, **_train_kwargs(p))
        trial.set_user_attr("best_epoch", epoch)
        return score

    name = f"FPMC ({axis})"
    study = optuna.create_study(
        study_name=name, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"{name}: best val {SELECT_METRIC} = {study.best_value:.4f} "
          f"({len(done)}/{n_trials} completed, {len(study.trials) - len(done)} pruned, "
          f"{time.time() - t0:.0f}s)")
    print(f"  params: {study.best_params}")
    print(f"  best epoch (E_best): {study.best_trial.user_attrs['best_epoch']}")
    return study


# ---------------------------------------------------------------------------
# Steps 3-5: refit on train+valid for E_best epochs, 5 seeds, score on test
# ---------------------------------------------------------------------------

def final_fpmc(data, params: dict, epochs: int, axis="persona", seeds=FINAL_SEEDS,
               loss: str = "ce", device: str = "cpu", verbose: bool = False):
    """Train one fresh model per seed on TRAIN + VALID for exactly `epochs`
    (= E_best, no early stopping) and score each on test.

    `params` = study.best_params. Returns (per_seed_df, models); report
    `per_seed_df.agg(["mean", "std"])` as Mean +/- Std.
    """
    import pandas as pd
    try:
        from src.preprocessing import merge_train_val
    except ImportError:                     # noqa
        from preprocessing import merge_train_val
    merged = merge_train_val(data)
    test_targets = [t["target_city"] for t in data["test"]]
    rows, models = [], []
    for seed in seeds:
        t0 = time.time()
        model, _, _ = train_fpmc(merged, axis=axis, loss=loss, seed=seed,
                                 fixed_epochs=epochs, device=device, verbose=verbose,
                                 **_train_kwargs(params))
        ranked = predict_ranked(model, data["test"], axis, device=device)
        m = evaluate(ranked, test_targets)
        m["train_secs"] = round(time.time() - t0, 1)
        m["n_params"] = sum(p.numel() for p in model.parameters())
        rows.append(m)
        models.append(model)
    df = pd.DataFrame(rows, index=[f"seed={s}" for s in seeds])
    print(f"\nFPMC ({axis}) - {len(seeds)} seeds")
    print(df.agg(["mean", "std"]).T.round(4).to_string())
    return df, models


def run_fpmc(data, axis="persona", device="cpu", **kw):
    """Quick single run (fit on train, early-stop on valid, predict test).
    Returns (metrics, ranked_lists, val_ndcg). The benchmark's final numbers come
    from `final_fpmc`, not from here."""
    t0 = time.time()
    model, val_score, _ = train_fpmc(data, axis=axis, device=device, **kw)
    ranked = predict_ranked(model, data["test"], axis, device=device)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics, ranked, val_score
