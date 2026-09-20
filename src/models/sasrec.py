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

Training is next-item prediction with a **BPR** loss over all valid positions
(each real step predicts the following city against one sampled negative) -- the
objective from the original SASRec paper, far cheaper than a 40k-way softmax.

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

try:
    from src.evaluate import evaluate
except ImportError:                          # noqa  (flat layout, e.g. Colab)
    from evaluate import evaluate

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
# Training (BPR over all valid positions) with early stopping on val NDCG@10
# ---------------------------------------------------------------------------

def train_sasrec(data, side_feats=SIDE_FEATURES, side_mode="sum", max_len=20,
                 dim=64, n_blocks=2, n_heads=2, dropout=0.2, epochs=40, lr=1e-3,
                 batch=256, seed=42, patience=5, device="cpu", verbose=True):
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

    model = SASRec(n_items, max_len, side_cards, side_feats, side_mode,
                   dim, n_blocks, n_heads, dropout).to(device)
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
            ndcg = evaluate(predict_ranked(model, data["valid"], device=device),
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


def run_sasrec(data, side_feats=SIDE_FEATURES, side_mode="sum", device="cpu", **kw):
    """Train, predict on test, and score. Returns (metrics, ranked_lists, val_ndcg)."""
    t0 = time.time()
    model, val_ndcg = train_sasrec(data, side_feats=side_feats,
                                   side_mode=side_mode, device=device, **kw)
    ranked = predict_ranked(model, data["test"], device=device)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics, ranked, val_ndcg
