"""
gru4rec.py -- GRU4Rec (model #3, "Ronia's")
=============================================

GRU4Rec compresses the whole trip prefix into a single recurrent hidden state
and scores every city from that one state, instead of pairwise transitions
(Markov/FPMC) or attending over the whole sequence (SASRec). It is the first
model in the project with genuine **long-range sequence memory**: capability
#3 on the sophistication ladder (Popularity -> Markov -> FPMC -> GRU4Rec ->
SASRec).

Architecture (built to sit on equal footing with SASRec -- the recurrence is the
only real difference between the two neural sequence models):

    city embedding * sqrt(dim)   (+ optional side-feature embeddings, trip-level
                                    context, broadcast and summed at every step)
      -> padded positions zeroed
      -> GRU (batch_first, bias-free so left-padding cannot change the state)
      -> linear head back into item-embedding space
      -> + residual connection from the input embedding   (optional, default on)
      -> LayerNorm; the last position's vector scores every candidate city.

Training is IDENTICAL to SASRec's: next-item prediction with a BPR loss over all
valid positions (each real step predicts the following city against one
uniformly sampled negative), the same left-padded / shifted-target examples,
the same Adam settings, and early stopping on validation NDCG@10.

Why these pieces (all taken from SASRec, none invented for GRU4Rec):

* sqrt(dim) embedding scale + LayerNorm -- SASRec uses both; without them the
  GRU sees tiny inputs (embedding std 0.02) and learns slowly.
* Residual connection -- SASRec's residual stream gives the output a direct
  path from the last city's embedding, i.e. an easy "last city -> next city"
  signal (which is most of the signal in this dataset: Markov, which sees only
  the last city, already scores well). A plain GRU has to learn that route
  through its gates. Set `residual=False` for a pure-GRU ablation.
* Padded steps are zeroed and the GRU has no bias, so a zero input keeps the
  hidden state exactly zero. The result is therefore independent of how much
  left-padding a trip has (SASRec achieves the same with its padding mask).
* Adam betas (0.9, 0.98), lr 1e-3, epochs ceiling 40, patience 5 -- same as
  SASRec.

Ablation the report requires: **with vs without side features** (device_class,
booker_country, affiliate_id), summed into the input embedding. Unlike
SASRec's separate sum/concat modes, GRU4Rec here only needs the on/off
comparison the plan asks for; the mechanism is deliberately the simpler,
already-validated "sum" mode from SASRec's ablation.

Every model in the project exposes the same output: a ranked list of city ids
per test trip, scored by the shared `evaluate.py`.
"""

from __future__ import annotations

import math
import time
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn

try:                                          # works both as a package and flat (Colab)
    from src.evaluate import evaluate
except ImportError:                           # noqa
    from evaluate import evaluate

PAD, UNK, N_RESERVED = 0, 1, 2
SIDE_FEATURES = ("device", "booker_country", "affiliate")   # trip-level context


# ---------------------------------------------------------------------------
# Sequence tensors (same left-pad / shifted-target scheme as SASRec, so the
# two neural models are trained on identical examples)
# ---------------------------------------------------------------------------

def _left_pad(seq: List[int], max_len: int) -> List[int]:
    """Keep the last `max_len` items, padding at the FRONT with PAD."""
    seq = seq[-max_len:]
    return [PAD] * (max_len - len(seq)) + seq


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
                 dim: int = 64, hidden: int = 64, n_layers: int = 1,
                 dropout: float = 0.2, residual: bool = True):
        super().__init__()
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
def predict_ranked(model, trips, max_len, k=100, batch=512, device="cpu"):
    model.eval()
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
# Training (BPR over all valid positions) with early stopping on val NDCG@10
# ---------------------------------------------------------------------------

def train_gru4rec(data, side_feats=(), max_len=20, dim=64, hidden=64, n_layers=1,
                  dropout=0.2, residual=True, epochs=40, lr=1e-3, batch=256,
                  seed=42, patience=5, device="cpu", verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    n_items = data["n_cities"]
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
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

    val_targets = [t["target_city"] for t in data["valid"]]
    best_ndcg, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            seq, tgt = X[idx], Y[idx]
            sd = {f: sides[f][idx] for f in side_feats}
            h = model.encode(seq, sd)
            neg = torch.randint(N_RESERVED, n_items, tgt.shape, device=device)
            pos_logit = (h * model.item_emb(tgt)).sum(-1)
            neg_logit = (h * model.item_emb(neg)).sum(-1)
            valid = (tgt != PAD).float()
            loss = -(torch.log(torch.sigmoid(pos_logit - neg_logit) + 1e-8)
                     * valid).sum() / valid.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

        ndcg = 0.0
        if val_targets:
            ndcg = evaluate(predict_ranked(model, data["valid"], max_len, device=device),
                            val_targets)["NDCG@10"]
        if verbose:
            print(f"  epoch {ep:2d}  loss={tot / n:.4f}  val_NDCG@10={ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg, bad = ndcg, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ndcg


def run_gru4rec(data, side_feats=(), max_len=20, device="cpu", **kw):
    """Train, predict on test, and score. Returns (metrics, ranked_lists, val_ndcg)."""
    t0 = time.time()
    model, val_ndcg = train_gru4rec(data, side_feats=side_feats, max_len=max_len,
                                    device=device, **kw)
    ranked = predict_ranked(model, data["test"], max_len, device=device)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics, ranked, val_ndcg