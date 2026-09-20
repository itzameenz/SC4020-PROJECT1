"""
fpmc.py -- Factorized Personalized Markov Chains (model #2, "ours")
==================================================================

FPMC fuses two ideas into one score for "given this traveller, sitting in city
`l`, how likely is city `i` next?":

    score(u, l, i) = < user_factor(u) , item_factor_U(i) >     # personalization
                   + < last_factor(l) , item_factor_L(i) >     # Markov transition

The first term is matrix factorization (who tends to like what); the second is a
*factorized* Markov chain (what usually follows what). Training uses **BPR**: for
an observed transition (u, l -> i) and a random non-next city j, push
score(u,l,i) above score(u,l,j).

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

try:                                        # works both as a package and flat (Colab)
    from src.evaluate import evaluate
except ImportError:                         # noqa
    from evaluate import evaluate

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
# Training (BPR) with early stopping on validation NDCG@10
# ---------------------------------------------------------------------------

def train_fpmc(data, axis="persona", dim=64, epochs=20, lr=1e-2, batch=4096,
               weight_decay=1e-6, seed=42, patience=3, device="cpu", verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
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
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_targets = [t["target_city"] for t in data["valid"]]
    best_ndcg, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            neg = torch.randint(N_RESERVED, n_items, (len(idx),), device=device)
            pos_s = model.score_pairs(u[idx], prev[idx], posb[idx], nxt[idx])
            neg_s = model.score_pairs(u[idx], prev[idx], posb[idx], neg)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

        ndcg = 0.0
        if val_targets:
            ranked = predict_ranked(model, data["valid"], axis, device=device)
            ndcg = evaluate(ranked, val_targets)["NDCG@10"]
        if verbose:
            print(f"  epoch {ep:2d}  loss={tot / n:.4f}  val_NDCG@10={ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg, best_state, bad = ndcg, {k: v.detach().clone()
                                                for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ndcg


def run_fpmc(data, axis="persona", device="cpu", **kw):
    """Train, predict on test, and score. Returns (metrics, ranked_lists, val_ndcg)."""
    t0 = time.time()
    model, val_ndcg = train_fpmc(data, axis=axis, device=device, **kw)
    ranked = predict_ranked(model, data["test"], axis, device=device)
    metrics = evaluate(ranked, [t["target_city"] for t in data["test"]])
    metrics["train_secs"] = round(time.time() - t0, 1)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics, ranked, val_ndcg
