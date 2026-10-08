"""
gru4rec.py -- GRU4Rec (model #3, "Ronia's")
=============================================

GRU4Rec compresses the whole trip prefix into a single recurrent hidden state
and scores every city from that one state, instead of pairwise transitions
(Markov/FPMC) or attending over the whole sequence (SASRec). It is the first
model in the project with genuine **long-range sequence memory**: capability
#3 on the sophistication ladder (Popularity -> Markov -> FPMC -> GRU4Rec ->
SASRec).

Architecture (kept from the Booking.com version -- the recurrence is the only
real difference to SASRec):

    city embedding * sqrt(dim)   (+ optional side-feature embeddings, trip-level
                                    context, broadcast and summed at every step)
      -> padded positions zeroed
      -> GRU (batch_first, bias-free so left-padding cannot change the state)
      -> linear head back into item-embedding space
      -> + residual connection from the input embedding   (optional, default on)
      -> LayerNorm; the last position's vector scores every candidate city
         against the (tied) item-embedding matrix.

Benchmark protocol (identical to models_benchmarking.ipynb)
-----------------------------------------------------------
* Loss: full-catalogue **cross-entropy** on the next city at every real step
  (`loss="ce"`). The old sampled-negative BPR is still available as
  `loss="bpr"` for ablation.
* Optimiser: **AdamW**, weight decay excluded from biases / norm weights
  (`build_adamw`), default betas. lr and weight_decay are tuned.
* Hidden size == embedding size (one `embed_dim` knob).
* Early stopping on validation NDCG@10: epoch ceiling `MAX_EPOCHS=20`,
  `PATIENCE=3`, best epoch restored.
* Search (`tune_gru4rec`): 25 Optuna TPE trials, `TPESampler(seed=42)`,
  `MedianPruner(n_warmup_steps=2)`, search seed 42, validation only:
      embed_dim {64,128,256} | num_layers {1,2} | dropout [0,0.5]
      lr [1e-4,1e-2] log     | weight_decay [1e-6,1e-3] log
      batch_size {64,128,256} | max_len {50,100,200}
* Final runs (`final_gru4rec`): record E_best of the winning trial, merge
  train + valid, train 5 fresh models (seeds 1, 7, 13, 2024, 99) for exactly
  E_best epochs, score on test, report mean +/- std.

Ablation the report requires: **with vs without side features** (device_class,
booker_country, affiliate_id), summed into the input embedding. Run
`tune_gru4rec` / `final_gru4rec` once with `side_feats=()` and once with
`side_feats=SIDE_FEATURES`; both get the identical search space and budget.

Padding: padded steps are zeroed and the GRU has no bias, so a zero input keeps
the hidden state exactly zero -- the result is independent of how much
left-padding a trip has. That is also why `max_len` is clipped to the longest
trip in the data (`_effective_len`): any larger value gives identical numbers
and only wastes compute.

Every model in the project exposes the same output: a ranked list of city ids
per test trip, scored by the shared `evaluate.py`.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                          # works both as a package and flat (Colab)
    from src.evaluate import evaluate
except ImportError:                           # noqa
    from evaluate import evaluate

PAD, UNK, N_RESERVED = 0, 1, 2
SIDE_FEATURES = ("device", "booker_country", "affiliate")   # trip-level context

# ---- benchmark protocol constants (match the notebook / every other model) ----
SELECT_METRIC = "NDCG@10"     # declared before any search
N_TRIALS = 25                 # hyper-parameter configurations per model
MAX_EPOCHS = 20               # ceiling on training length within one trial
PATIENCE = 3                  # stop after 3 epochs without NDCG@10 improvement
SEARCH_SEED = 42              # fixed during search so trials differ only in hyper-params
FINAL_SEEDS = [1, 7, 13, 2024, 99]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_adamw(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with decay excluded from biases and normalisation weights."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "norm" in name.lower() else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr)


# ---------------------------------------------------------------------------
# Sequence tensors (left-pad / shifted-target scheme shared with SASRec)
# ---------------------------------------------------------------------------

def _left_pad(seq: List[int], max_len: int) -> List[int]:
    """Keep the last `max_len` items, padding at the FRONT with PAD."""
    seq = seq[-max_len:]
    return [PAD] * (max_len - len(seq)) + seq


def _effective_len(data, max_len: int) -> int:
    """Clip `max_len` to the longest trip in any split. Results are identical
    (padding is inert, see module docstring); it only avoids allocating [B, 200]
    tensors for trips that are a few cities long."""
    longest = max((len(t["cities"]) for k in ("train", "valid", "test")
                   for t in data.get(k, [])), default=1)
    return max(1, min(max_len, longest))


def build_matrix(trips, max_len: int, training: bool):
    """Return input/target id matrices [N, max_len] plus side-feature ids [N].

    Training: input = cities[:-1], target = cities[1:] (predict the next city
    at every position). Inference: input = the observed prefix, target unused.
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
# Model
# ---------------------------------------------------------------------------

class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, side_cards=None, side_feats: Sequence[str] = (),
                 dim: int = 64, hidden: int | None = None, n_layers: int = 1,
                 dropout: float = 0.2, residual: bool = True):
        super().__init__()
        hidden = hidden or dim                  # benchmark: hidden size == embedding size
        self.dim = dim
        self.side_feats = tuple(side_feats)
        self.residual = residual

        self.item_emb = nn.Embedding(n_items, dim, padding_idx=PAD)
        if self.side_feats:
            self.side_emb = nn.ModuleDict(
                {f: nn.Embedding(side_cards[f], dim) for f in self.side_feats})

        # bias=False: with zero input at padded steps the hidden state stays exactly
        # zero (update/reset gates are 0.5, candidate is tanh(0) = 0), so left-padding
        # cannot leak into the state.
        self.gru = nn.GRU(dim, hidden, num_layers=n_layers, batch_first=True,
                          bias=False, dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, dim)   # project hidden state -> item-embedding space
        self.norm = nn.LayerNorm(dim)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()
        if self.side_feats:
            for f in self.side_feats:
                nn.init.normal_(self.side_emb[f].weight, std=0.02)

    def encode(self, seq, sides):
        """[B, L] city ids -> [B, L, dim] query vectors, one per step."""
        B, L = seq.shape
        real = (seq != PAD).unsqueeze(-1).float()                 # [B, L, 1]
        x = self.item_emb(seq) * math.sqrt(self.dim)
        if self.side_feats:
            for f in self.side_feats:
                x = x + self.side_emb[f](sides[f]).unsqueeze(1).expand(B, L, -1)
        x = x * real                                              # zero padded steps
        x = self.drop(x)
        h, _ = self.gru(x)
        z = self.out(h)
        if self.residual:
            z = z + x                                             # direct last-city path
        return self.norm(z)

    def score_last(self, seq, sides):
        """Scores over all items from the final position -> [B, n_items]."""
        return self.encode(seq, sides)[:, -1, :] @ self.item_emb.weight.t()


# ---------------------------------------------------------------------------
# Prediction -> ranked lists
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_ranked(model, trips, max_len=None, k=100, batch=512, device="cpu"):
    model.eval()
    max_len = max_len or model.max_len
    X, _, sides = build_matrix(trips, max_len, training=False)
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

def train_gru4rec(data, side_feats=(), max_len=50, dim=64, hidden=None, n_layers=1,
                  dropout=0.2, residual=True, loss="ce", epochs=MAX_EPOCHS, lr=1e-3,
                  batch=256, weight_decay=0.0, seed=SEARCH_SEED, patience=PATIENCE,
                  fixed_epochs=None, select_metric=SELECT_METRIC, trial=None,
                  device="cpu", verbose=True):
    """Train one GRU4Rec. Returns (model, best validation score, best epoch).

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

    model = GRU4Rec(n_items, side_cards, side_feats, dim, hidden, n_layers,
                    dropout, residual).to(device)
    model.max_len = max_len
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
        "num_layers":   trial.suggest_categorical("num_layers", [1, 2]),
        "batch_size":   trial.suggest_categorical("batch_size", [64, 128, 256]),
        "max_len":      trial.suggest_categorical("max_len", [10, 20, 30, 40]),
    }


def _train_kwargs(p: dict) -> dict:
    """Optuna param names -> train_gru4rec arguments."""
    return dict(max_len=p["max_len"], dim=p["embed_dim"], n_layers=p["num_layers"],
                dropout=p["dropout"], lr=p["lr"], batch=p["batch_size"],
                weight_decay=p["weight_decay"])


def tune_gru4rec(data, side_feats=(), n_trials: int = N_TRIALS, seed: int = SEARCH_SEED,
                 residual: bool = True, loss: str = "ce", device: str = "cpu",
                 show_progress_bar: bool = True):
    """Optuna search over the unified space on validation NDCG@10. Returns the
    finished `optuna.Study`; `study.best_params` is the winning config and
    `study.best_trial.user_attrs["best_epoch"]` is E_best."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = suggest_params(trial)
        _, score, epoch = train_gru4rec(data, side_feats=side_feats, residual=residual,
                                        loss=loss, trial=trial, device=device,
                                        verbose=False, **_train_kwargs(p))
        trial.set_user_attr("best_epoch", epoch)
        return score

    name = f"GRU4Rec ({'side' if side_feats else 'no side'})"
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

def final_gru4rec(data, params: dict, epochs: int, side_feats=(), seeds=FINAL_SEEDS,
                  residual: bool = True, loss: str = "ce", device: str = "cpu",
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
        model, _, _ = train_gru4rec(merged, side_feats=side_feats, residual=residual,
                                    loss=loss, seed=seed, fixed_epochs=epochs,
                                    device=device, verbose=verbose, **_train_kwargs(params))
        ranked = predict_ranked(model, data["test"], device=device)
        m = evaluate(ranked, test_targets)
        m["train_secs"] = round(time.time() - t0, 1)
        m["n_params"] = sum(p.numel() for p in model.parameters())
        rows.append(m)
        models.append(model)
    df = pd.DataFrame(rows, index=[f"seed={s}" for s in seeds])
    print(f"\nGRU4Rec ({'side' if side_feats else 'no side'}) - {len(seeds)} seeds")
    print(df.agg(["mean", "std"]).T.round(4).to_string())
    return df, models


def run_gru4rec(data, side_feats=(), device="cpu", **kw):
    """Quick single run (fit on train, early-stop on valid, predict test).
    Returns (metrics, ranked_lists, val_ndcg). The benchmark's final numbers come
    from `final_gru4rec`, not from here."""
    t0 = time.time()
    model, val_score, _ = train_gru4rec(data, side_feats=side_feats, device=device, **kw)
    ranked = predict_ranked(model, data["test"], device=device)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics, ranked, val_score
