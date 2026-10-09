"""
sasrec.py -- Self-Attentive Sequential Recommendation (model #4, "ours")
=======================================================================

SASRec replaces a recurrent hidden state with **causal self-attention** over the
city sequence: at each step it learns which earlier stops matter most for
predicting the next one, instead of compressing everything equally. It is the
natural place to "bolt on" side features.

Architecture (a standard SASRec, hand-rolled so the causal + padding masks are
fully under our control -- PyTorch's built-in TransformerEncoder produced NaN-
collapsed attention on fully-masked padding rows):

    city embedding * sqrt(dim)  +  learned positional embedding
      (+ optional side-feature embeddings: device_class, booker_country, affiliate)
    -> N causal self-attention blocks (pre-norm, residual, GELU feed-forward)
    -> final LayerNorm; the last position's vector scores every candidate city.

Benchmark protocol (identical to gru4rec.py -- constants are imported from there)
--------------------------------------------------------------------------------
* Loss: full-catalogue **cross-entropy** on the next city at every real step
  (`loss="ce"`), same as GRU4Rec. The original paper's sampled-negative BPR is
  still available as `loss="bpr"`.
* Optimiser: **AdamW** (`build_adamw`, decay excluded from biases / norm
  weights). lr and weight_decay are tuned.
* Early stopping on validation NDCG@10: `MAX_EPOCHS=20`, `PATIENCE=3`.
* Search (`tune_sasrec`): 25 Optuna TPE trials, `TPESampler(seed=42)`,
  `MedianPruner(n_warmup_steps=2)`, validation only:
      embed_dim {64,128,256} | num_blocks {1,2}     | dropout [0,0.5]
      lr [1e-4,1e-2] log     | weight_decay [1e-6,1e-3] log
      batch_size {64,128,256} | max_len {10,20,30,40}
  `n_heads` is fixed at 2 (every embed_dim choice divides by it).
* Final runs (`final_sasrec`): record E_best of the winning trial, merge
  train + valid, train 5 fresh models (seeds 1, 7, 13, 2024, 99) for exactly
  E_best epochs, score on test, report mean +/- std.

Ablations the report requires:
* side features on/off (which ones move the metrics),
* how they combine -- `side_mode="sum"` (add) vs `side_mode="concat"` (concat then
  project back to the model width).

Note for the write-up: transformers are data-hungry. On a small subsample with
short trips, SASRec can trail the simpler models because the signal is dominated
by the last-city transition; its advantage is expected to appear on the full
dataset (train on a Colab GPU). Output matches every model: a ranked list of city
ids per test trip.
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.evaluate import evaluate
    from src.models.gru4rec import (set_seed, build_adamw, _effective_len, SELECT_METRIC,
                                    N_TRIALS, MAX_EPOCHS, PATIENCE, SEARCH_SEED, FINAL_SEEDS)
except ImportError:                          # noqa  (flat layout, e.g. Colab)
    from evaluate import evaluate
    from gru4rec import (set_seed, build_adamw, _effective_len, SELECT_METRIC,
                         N_TRIALS, MAX_EPOCHS, PATIENCE, SEARCH_SEED, FINAL_SEEDS)

PAD, UNK, N_RESERVED = 0, 1, 2
SIDE_FEATURES = ("device", "booker_country", "affiliate")   # trip-level context


# ---------------------------------------------------------------------------
# Sequence tensors
# ---------------------------------------------------------------------------

def _left_pad(seq: List[int], max_len: int) -> List[int]:
    """Keep the last `max_len` items, padding at the FRONT with PAD."""
    seq = seq[-max_len:]
    return [PAD] * (max_len - len(seq)) + seq


def build_matrix(trips, max_len: int, training: bool):
    """Return input/target id matrices [N, max_len] plus side-feature ids [N].

    Training: input = cities[:-1], target = cities[1:] (predict the next city at
    every position). Inference: input = the observed prefix, target unused.
    """
    X, Y, sides = [], [], {f: [] for f in SIDE_FEATURES}
    for t in trips:
        cities = t["cities"]
        inp, tgt = (cities[:-1], cities[1:]) if training else (cities, [])
        if len(inp) == 0:
            continue
        X.append(_left_pad(inp, max_len))
        Y.append(_left_pad(tgt, max_len) if training else [PAD] * max_len)
        for f in SIDE_FEATURES:
            sides[f].append(t[f])
    X = np.asarray(X, dtype=np.int64)
    Y = np.asarray(Y, dtype=np.int64)
    sides = {f: np.asarray(v, dtype=np.int64) for f, v in sides.items()}
    return X, Y, sides


# ---------------------------------------------------------------------------
# One causal self-attention block (pre-norm, residual)
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dk = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal, pad):
        B, L, d = x.shape
        y = self.ln1(x)
        q = self.q(y).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(y).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(y).view(B, L, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(causal[None, None], float("-inf"))
        att = att.masked_fill(pad[:, None, None, :], float("-inf"))
        att = torch.nan_to_num(torch.softmax(att, dim=-1))   # zero fully-masked rows
        z = (att @ v).transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(self.o(z))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SASRec(nn.Module):
    def __init__(self, n_items, max_len, side_cards=None, side_feats=(),
                 side_mode="sum", dim=64, n_blocks=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.dim, self.max_len = dim, max_len
        self.side_feats = tuple(side_feats)
        self.side_mode = side_mode if self.side_feats else "none"

        self.item_emb = nn.Embedding(n_items, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)

        if self.side_feats:
            sd = dim if self.side_mode == "sum" else dim // 2
            self.side_emb = nn.ModuleDict(
                {f: nn.Embedding(side_cards[f], sd) for f in self.side_feats})
            if self.side_mode == "concat":
                self.side_proj = nn.Linear(dim + sd * len(self.side_feats), dim)

        self.blocks = nn.ModuleList(
            [_Block(dim, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()
        if self.side_feats:
            for f in self.side_feats:
                nn.init.normal_(self.side_emb[f].weight, std=0.02)

    def encode(self, seq, sides):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) * math.sqrt(self.dim) + self.pos_emb(pos)
        if self.side_feats:
            vecs = [self.side_emb[f](sides[f]).unsqueeze(1).expand(B, L, -1)
                    for f in self.side_feats]
            if self.side_mode == "sum":
                for vv in vecs:
                    x = x + vv
            else:  # concat then project back to dim
                x = self.side_proj(torch.cat([x] + vecs, dim=-1))
        x = self.drop(x)
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool),
                            diagonal=1)
        pad = (seq == PAD)
        for blk in self.blocks:
            x = blk(x, causal, pad)
        return self.norm(x)

    def score_last(self, seq, sides):
        """Scores over all items from the final position -> [B, n_items]."""
        return self.encode(seq, sides)[:, -1, :] @ self.item_emb.weight.t()


# ---------------------------------------------------------------------------
# Prediction -> ranked lists
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_ranked(model, trips, k=100, batch=512, device="cpu"):
    model.eval()
    X, _, sides = build_matrix(trips, model.max_len, training=False)
    ranked: List[list] = []
    for s in range(0, len(X), batch):
        seq = torch.tensor(X[s:s + batch], device=device)
        sd = {f: torch.tensor(sides[f][s:s + batch], device=device)
              for f in model.side_feats}
        scores = model.score_last(seq, sd)
        scores[:, PAD] = -np.inf
        scores[:, UNK] = -np.inf
        ranked.extend(torch.topk(scores, k, dim=1).indices.cpu().tolist())
    return ranked


# ---------------------------------------------------------------------------
# Training with early stopping on val NDCG@10
# ---------------------------------------------------------------------------

def train_sasrec(data, side_feats=SIDE_FEATURES, side_mode="sum", max_len=20,
                 dim=64, n_blocks=2, n_heads=2, dropout=0.2, loss="ce",
                 epochs=MAX_EPOCHS, lr=1e-3, batch=256, weight_decay=0.0,
                 seed=SEARCH_SEED, patience=PATIENCE, fixed_epochs=None,
                 select_metric=SELECT_METRIC, trial=None, device="cpu", verbose=True):
    """Train one SASRec. Returns (model, best validation score, best epoch).

    `fixed_epochs` => final-fit mode: train exactly that many epochs on
    data['train'] (pass the train+valid merge), no validation, no early stopping,
    keep the LAST-epoch weights; the returned score is NaN.
    `trial` => an optuna.Trial: report val score each epoch and allow pruning.
    """
    set_seed(seed)
    n_items = data["n_cities"]
    max_len = _effective_len(data, max_len)
    side_cards = {
        "device": len(data["encoders"]["device_class"]) + N_RESERVED,
        "booker_country": len(data["encoders"]["booker_country"]) + N_RESERVED,
        "affiliate": len(data["encoders"]["affiliate_id"]) + N_RESERVED,
    }

    X, Y, sides = build_matrix(data["train"], max_len, training=True)
    X = torch.tensor(X, device=device); Y = torch.tensor(Y, device=device)
    sides = {f: torch.tensor(v, device=device) for f, v in sides.items()}
    n = len(X)

    model = SASRec(n_items, max_len, side_cards, side_feats, side_mode,
                   dim, n_blocks, n_heads, dropout).to(device)
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
            seq, tgt = X[idx], Y[idx]
            sd = {f: sides[f][idx] for f in side_feats}
            h = model.encode(seq, sd)                              # [B, L, dim]
            valid = tgt != PAD
            if loss == "ce":
                # full-catalogue cross-entropy, computed on real steps only so the
                # logits tensor is [n_real_steps, n_items], not [B*L, n_items]
                logits = h[valid] @ model.item_emb.weight.t()
                l = F.cross_entropy(logits, tgt[valid])
            else:                                                  # "bpr": 1 sampled negative
                neg = torch.randint(N_RESERVED, n_items, tgt.shape, device=device)
                pos_logit = (h * model.item_emb(tgt)).sum(-1)
                neg_logit = (h * model.item_emb(neg)).sum(-1)
                v = valid.float()
                l = -(torch.log(torch.sigmoid(pos_logit - neg_logit) + 1e-8) * v).sum() / v.sum()
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item() * len(idx)

        if fixed_epochs is not None:
            if verbose:
                print(f"  epoch {ep:2d}  loss={tot / n:.4f}  (fixed-epoch final fit)")
            continue

        score = 0.0
        if val_targets:
            score = evaluate(predict_ranked(model, data["valid"], device=device),
                             val_targets)[select_metric]
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
    """Unified search space shared by every neural model in the benchmark."""
    return {
        "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout":      trial.suggest_float("dropout", 0.0, 0.5),
        "embed_dim":    trial.suggest_categorical("embed_dim", [64, 128, 256]),
        "num_blocks":   trial.suggest_categorical("num_blocks", [1, 2]),
        "batch_size":   trial.suggest_categorical("batch_size", [64, 128, 256]),
        "max_len":      trial.suggest_categorical("max_len", [10, 20, 30, 40]),
    }


def _train_kwargs(p: dict) -> dict:
    """Optuna param names -> train_sasrec arguments."""
    return dict(max_len=p["max_len"], dim=p["embed_dim"], n_blocks=p["num_blocks"],
                dropout=p["dropout"], lr=p["lr"], batch=p["batch_size"],
                weight_decay=p["weight_decay"])


def _variant_name(side_feats, side_mode) -> str:
    return f"SASRec ({side_mode if side_feats else 'no side'})"


def tune_sasrec(data, side_feats=(), side_mode="sum", n_trials: int = N_TRIALS,
                seed: int = SEARCH_SEED, loss: str = "ce", device: str = "cpu",
                show_progress_bar: bool = True):
    """Optuna search over the unified space on validation NDCG@10. Returns the
    finished `optuna.Study`; `study.best_params` is the winning config and
    `study.best_trial.user_attrs["best_epoch"]` is E_best."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = suggest_params(trial)
        _, score, epoch = train_sasrec(data, side_feats=side_feats, side_mode=side_mode,
                                       loss=loss, trial=trial, device=device,
                                       verbose=False, **_train_kwargs(p))
        trial.set_user_attr("best_epoch", epoch)
        return score

    name = _variant_name(side_feats, side_mode)
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

def final_sasrec(data, params: dict, epochs: int, side_feats=(), side_mode="sum",
                 seeds=FINAL_SEEDS, loss: str = "ce", device: str = "cpu",
                 verbose: bool = False):
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
        model, _, _ = train_sasrec(merged, side_feats=side_feats, side_mode=side_mode,
                                   loss=loss, seed=seed, fixed_epochs=epochs,
                                   device=device, verbose=verbose, **_train_kwargs(params))
        ranked = predict_ranked(model, data["test"], device=device)
        m = evaluate(ranked, test_targets)
        m["train_secs"] = round(time.time() - t0, 1)
        m["n_params"] = sum(p.numel() for p in model.parameters())
        rows.append(m)
        models.append(model)
    df = pd.DataFrame(rows, index=[f"seed={s}" for s in seeds])
    print(f"\n{_variant_name(side_feats, side_mode)} - {len(seeds)} seeds")
    print(df.agg(["mean", "std"]).T.round(4).to_string())
    return df, models


def run_sasrec(data, side_feats=SIDE_FEATURES, side_mode="sum", device="cpu", **kw):
    """Quick single run (fit on train, early-stop on valid, predict test).
    Returns (metrics, ranked_lists, val_ndcg). The benchmark's final numbers come
    from `final_sasrec`, not from here."""
    t0 = time.time()
    model, val_score, _ = train_sasrec(data, side_feats=side_feats,
                                       side_mode=side_mode, device=device, **kw)
    ranked = predict_ranked(model, data["test"], device=device)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics, ranked, val_score
