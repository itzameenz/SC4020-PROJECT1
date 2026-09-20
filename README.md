# SC4020 Project 1 — Sequential Recommendation on the Booking.com MDT Dataset

A detailed, self-contained writeup of the codebase: what the task is, how the data
is prepared, how models are evaluated, how our two models (FPMC and SASRec) work,
and what the results mean. Written so a teammate can read it cold, understand every
file, reproduce every number, and plug in their own model. It doubles as raw
material for the report.

> **One-sentence summary.** We predict a traveller's next city in a multi-city
> trip, comparing five models of increasing sophistication (Popularity → Markov →
> FPMC → GRU4Rec → SASRec) through a single shared, leakage-free pipeline, so that
> differences in the scores reflect the *models* rather than differences in how the
> data was prepared or scored.

## Table of contents
1. [The problem](#1-the-problem)
2. [The dataset and the given split](#2-the-dataset-and-the-given-split)
3. [Design principle: one shared pipeline](#3-design-principle-one-shared-pipeline)
4. [Preprocessing, step by step](#4-preprocessing-step-by-step)
5. [Evaluation, in detail](#5-evaluation-in-detail)
6. [Model 1 — FPMC](#6-model-1--fpmc)
7. [Model 2 — SASRec](#7-model-2--sasrec)
8. [Results and analysis](#8-results-and-analysis)
9. [File-by-file reference](#9-file-by-file-reference)
10. [For the teammate: plug in your own model](#10-for-the-teammate-plug-in-your-own-model)
11. [How to run](#11-how-to-run)
12. [Submission constraints](#12-submission-constraints)

---

## 1. The problem

Booking.com customers often book several cities in one trip. Given the cities a
traveller has already booked **in order**, we want to **predict the next city**.
This is a *sequential recommendation* (next-item prediction) problem: the "session"
is a trip, and the "items" are cities.

Why it is interesting: the right next city depends on where the traveller just was
(a strong local transition, e.g. nearby cities), on longer-range structure (a
multi-stop route), and possibly on context (which country they book from, device,
season). Different models capture different amounts of this, and our experiments
show *where each additional capability actually pays off*.

## 2. The dataset and the given split

**Booking.com Multi-Destination Trips (MDT).** Source:
<https://github.com/bookingcom/ml-dataset-mdt>. Three CSVs go into `data/`:

| File | Rows | Role |
|---|---|---|
| `train_set.csv` | ~1.17M reservations | full trips — training |
| `test_set.csv` | ~0.28M reservations | each trip's **final city masked** (`city_id = 0`) |
| `ground_truth.csv` | ~70.7k trips | the true masked city per test trip — scoring only |

**Columns.** `utrip_id` (trip id, groups a trip's rows), `user_id`, `checkin`,
`checkout`, `city_id` (**the item we predict**), `hotel_country`, `booker_country`,
`device_class`, `affiliate_id`.

**How the split works — this *is* the leave-last-out protocol.** For each test
trip, the dataset **hides the last stop**: that row is still present but its
`city_id` is set to `0` and `hotel_country` is blank. The model therefore sees the
whole trip *up to* the final stop and must predict the final city; the true answer
lives in `ground_truth.csv`. Because the dataset ships this masking, we **reuse it
directly** and never construct our own test split — this also makes our numbers
comparable to the public Booking.com challenge.

*(The assignment requires ≥2 datasets; MDT is dataset #1, and a second-domain
dataset is handled separately by the wider group.)*

## 3. Design principle: one shared pipeline

The credibility of the whole comparison rests on one idea: **everything that is
shared must be identical for all five models.** Two files carry that:

- `src/preprocessing.py` — raw CSVs → clean, encoded sequences + the train/val/test
  split.
- `src/evaluate.py` — turns any model's predictions into the same metrics.

They meet the models through a single **interface**: every model, however it works
inside, outputs a **ranked list of city ids per test trip** (best guess first). The
scorer only looks at where the true city lands in that list. This is why a
frequency count, a Markov table, and a transformer can all appear in one results
table and be compared fairly.

```
raw CSVs ──preprocessing.py──▶ clean sequences + features + splits
                                        │
              ┌───────────┬─────────────┼─────────────┬───────────┐
          Popularity    Markov        FPMC         GRU4Rec      SASRec
              │           │             │              │           │
              └───────────┴──── ranked list per trip ──┴───────────┘
                                        │
                                 evaluate.py ──▶ HR@k, NDCG@k, MRR, ...
```

## 4. Preprocessing, step by step

All of this lives in `src/preprocessing.py` and is applied **identically** to every
model. Each step, what it does, and *why*:

1. **Clean.** Drop exact-duplicate reservation rows and rows whose check-in date
   cannot be parsed (they can't be ordered). This is deliberately conservative so it
   never corrupts a sequence; `checkout < checkin` is handled by clamping the
   length-of-stay feature to ≥ 0 rather than deleting the row.

2. **Order.** Within each `utrip_id`, sort rows by check-in date to recover the true
   visit order — the sequence the models learn from.

3. **Collapse consecutive duplicates.** If the same city appears in two back-to-back
   rows, that is one longer stay, not a move to a new destination. We collapse it, so
   the models learn genuine city→city transitions instead of a trivial self-loop that
   would otherwise inflate every metric.

4. **Minimum trip length.** Drop **training** trips with fewer than 4 stops, to match
   the test set (whose trips all have ≥ 4). Length-1 trips are always dropped (nothing
   to predict). The **test set is never filtered**. `min_trip_len` is a parameter, so
   this is revisitable as an ablation.

5. **Categorical encoding.** Map each categorical (`city_id`, `user_id`,
   `hotel_country`, `booker_country`, `device_class`, `affiliate_id`) to a contiguous
   integer using a mapping built **only from the training set**. Two indices are
   reserved: **`0 = PAD`** (padding, used later by the neural models' data loaders)
   and **`1 = UNK`** (a value unseen in training); real ids start at 2. Keeping PAD
   and UNK separate matters for GRU4Rec/SASRec — a padded position must not look like a
   genuine unknown city.

6. **Engineered features** (a superset; each model uses only what it needs):
   - `month` of check-in (1–12) — seasonality,
   - `length_of_stay` (nights) — trip intent,
   - trip **position** as an index, a 0–1 fraction, and a 5-bucket version — mainly for
     FPMC, which (unlike SASRec) has no built-in notion of position,
   - a `persona` = (`booker_country`, `device_class`) — a coarse profile that recurs
     across trips, for FPMC's personalization ablation.

7. **Three-way, leakage-free split.**
   - **train** — 90% of (filtered) train trips, full sequences → fit the models,
   - **validation** — 10% of train trips with the **last city hidden** → tune
     hyperparameters + early-stopping,
   - **test** — the given masked test set + ground-truth target → touched only for
     final numbers.
   Validation is carved out of train so the neural models can be tuned **without ever
   seeing the test set** — the single most important guard against optimistic results.

**Leakage guards:** encoders are fit on train only; validation comes from train only;
only pre-decision context is ever fed to a model. **What is *not* here** (on purpose):
padding/truncation to a fixed length, and each model's training-example construction
(Markov counts, FPMC triples, SASRec shifted sequences) live inside the models,
because different models need different formats.

**Output.** A single dict, pickled to `data/processed/processed.pkl`, with keys
`train`, `valid`, `test`, `encoders`, `persona_encoder`, `n_cities`, `meta`. On the
full data: **190,737 train / 21,187 validation / 70,662 test trips, 39,903 cities.**

## 5. Evaluation, in detail

All of this lives in `src/evaluate.py`.

**Protocol.** Leave-last-out, exactly as the dataset ships it: hide the last city,
rank all cities, check where the true city landed.

**Metrics.** For a trip whose true city ends up at rank *r* (1-based) in the model's
ranked list:
- **HR@k** (Hit Rate) = 1 if *r* ≤ k else 0 — *"is the true city in the top k?"*
- **NDCG@k** = 1 / log₂(*r* + 1) if *r* ≤ k else 0 — *"how high up is it?"* (a single
  relevant item means the ideal DCG is 1).
- **MRR** (Mean Reciprocal Rank) = 1 / *r* (0 if not found) — *"how good is its
  position on average?"*

All are averaged over the 70,662 test trips and reported for **k ∈ {1, 5, 10}**,
ranking against the **full 39,903-city catalog**. The reserved ids `0` (PAD) and `1`
(UNK) are never valid predictions; a trip whose true city was unseen in training maps
to UNK and is an automatic miss (as it would be in deployment).

**Correctness peg.** We also compute **Accuracy@4** (= HR@4). Computed the repo's way
(raw city counts), the Popularity baseline reproduces the repo's demo value **0.0527
≈ 0.053**, confirming the scorer is implemented correctly.

**Deeper analyses.** `evaluate_sliced(...)` reports metrics per subgroup (e.g. by
trip length, or by destination popularity via `city_frequency(...)`), which is how the
report turns a leaderboard into findings.

**The interface a model must satisfy:**
```python
from evaluate import evaluate, append_result
targets = [t["target_city"] for t in data["test"]]
metrics = evaluate(ranked_lists, targets)     # ranked_lists aligned to data["test"]
append_result("results.csv", "MyModel", metrics)
```

## 6. Model 1 — FPMC

**Idea.** FPMC (Factorized Personalized Markov Chains) scores *"given this traveller,
currently in city `l`, how likely is city `i` next?"* by adding two factorized terms:

```
score(u, l, i) = < user_factor(u) , item_factor_U(i) >     # personalization (MF)
               + < last_factor(l) , item_factor_L(i) >     # transition (Markov)
```

The first term is matrix factorization (who tends to prefer what); the second is a
*factorized* Markov chain (what usually follows what). We add a **position term**: a
position-bucket embedding added to the user side, so FPMC can express "early in the
trip vs the leg home".

**Training.** Bayesian Personalized Ranking (**BPR**): for an observed transition
`l → i` and a random city `j`, maximise `sigmoid(score(u,l,i) − score(u,l,j))`. Trained
with Adam, early-stopped on validation NDCG@10.

**Ablation — the personalization axis.** We run FPMC two ways:
- `axis="user"` — raw `user_id`,
- `axis="persona"` — (booker_country, device_class).
Because ~200k users mostly appear in a single trip, the raw per-user factor is cold at
test time, whereas the persona recurs across trips. On the full data **persona
slightly beats user (NDCG@10 0.350 vs 0.345)** — the hypothesised result and a headline
finding. (Interesting nuance: on a small subsample the ordering flips, because there
the few repeated users help more than the coarse persona.)

**Training behaviour to discuss.** FPMC's validation NDCG peaks at **epoch 1** and then
declines — it learns the transition structure almost instantly and overfits quickly.
Early stopping captures the best checkpoint, so the reported numbers are valid, but a
lower learning rate + more regularization is the natural tuning direction (good
"parameter settings" content for the report). Code: `src/models/fpmc.py`; notebook:
`notebooks/04_fpmc.ipynb`.

## 7. Model 2 — SASRec

**Idea.** SASRec (Self-Attentive Sequential Recommendation) replaces a recurrent
hidden state with **causal self-attention** over the city sequence: at each step it
learns *which earlier stops matter most* for predicting the next one, instead of
compressing everything equally.

**Architecture** (`src/models/sasrec.py`):
```
city embedding × sqrt(dim)  +  learned positional embedding
  (+ optional side-feature embeddings: device_class, booker_country, affiliate)
→ N causal self-attention blocks (pre-norm, residual, GELU feed-forward)
→ final LayerNorm; the last position's vector scores every candidate city.
```
Trained with the same BPR next-item objective over all valid positions (each real step
predicts the following city against one sampled negative), early-stopped on validation
NDCG@10.

**Implementation note (worth mentioning in the report).** Our first version used
PyTorch's built-in `nn.TransformerEncoder`, which *collapsed*: with the padding mask,
fully-masked positions produced NaNs that degenerate the attention, so every trip got
the same near-constant prediction (NDCG ≈ 0.004). We replaced it with a **hand-rolled
causal-attention block** that explicitly zeroes fully-masked rows, which fixed it and
made SASRec competitive. This is a genuine debugging finding, not a shortcut.

**Ablation — side features.** We compare none vs **summed** vs **concatenated** (then
projected) trip-level context (device, booker_country, affiliate). Result: side
features **hurt** here (no-side 0.351 > sum 0.332 > concat 0.290 NDCG@10) — the
trip-level context is too coarse/noisy for this task, and concatenation's extra
parameters hurt most. Notebook: `notebooks/05_sasrec.ipynb`.

**Compute note.** SASRec is data- and compute-hungry: on CPU each full-data config
took ~13–17 minutes; the proper setting is a Colab GPU (a few minutes). FPMC, being
shallow, trains in ~1–2 minutes on CPU.

## 8. Results and analysis

**Full dataset (190,737 train / 21,187 val / 70,662 test trips, 39,903 cities):**

| Model | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR | Accuracy@4 |
|---|---|---|---|---|---|---|---|
| Popularity | 0.018 | 0.052 | 0.102 | 0.034 | 0.050 | 0.035 | 0.043 |
| FPMC-persona | 0.192 | 0.432 | 0.531 | 0.318 | **0.350** | 0.294 | 0.397 |
| FPMC-user | 0.192 | 0.428 | 0.515 | 0.316 | 0.345 | 0.291 | 0.395 |
| SASRec-no-side | 0.183 | 0.434 | **0.549** | 0.313 | **0.351** | 0.289 | 0.396 |
| SASRec-side-sum | 0.158 | 0.416 | 0.537 | 0.292 | 0.332 | 0.268 | 0.378 |
| SASRec-side-concat | 0.131 | 0.360 | 0.494 | 0.247 | 0.290 | 0.228 | 0.316 |

*(rows produced by `results.csv`; Markov + GRU4Rec are added by the teammate)*

**What the numbers say:**
1. **Both real models beat the Popularity floor by ~7×** on NDCG@10 (0.35 vs 0.05) —
   sequence/transition information is highly predictive of the next city.
2. **FPMC-persona ≥ FPMC-user** on full data — evidence that with mostly-single-trip
   users the coarse persona generalises where a raw per-user factor cannot. This is
   the payoff of the personalization ablation.
3. **SASRec's side features hurt** (no-side > sum > concat) — a clean negative
   ablation result: the trip-level context here adds noise, not signal.
4. **SASRec (no-side) and FPMC-persona are neck-and-neck** — SASRec wins HR@10
   (0.549 vs 0.531), FPMC-persona ties/edges NDCG@10. On this dataset, with short
   trips dominated by the last-city transition, the transformer's extra machinery buys
   little over a well-built factorized Markov model — itself a worthwhile conclusion.
5. **Cold-tail problem** (from `06`'s sliced analysis): models score strongly on
   popular (head) destinations but near-zero on rare (tail) cities.

**Sanity peg.** Popularity's Accuracy@4 = 0.0527 (repo-style raw counts) reproduces the
repo's ~0.053 → the evaluation code is trustworthy.

## 9. File-by-file reference

```
project1/
├── data/                        raw CSVs (gitignored) + processed/processed.pkl
├── src/
│   ├── preprocessing.py         SHARED — clean/encode/feature/split (Section 4)
│   ├── evaluate.py              SHARED — metrics + sliced analysis (Section 5)
│   └── models/
│       ├── fpmc.py              FPMC: build_transitions, FPMC(nn.Module), train_fpmc, run_fpmc
│       └── sasrec.py            SASRec: build_matrix, SASRec(nn.Module), train_sasrec, run_sasrec
├── notebooks/
│   ├── 01_data_exploration.ipynb   EDA — trip lengths, user sparsity, seasonality
│   ├── 02_preprocessing.ipynb      runs preprocessing, writes processed.pkl
│   ├── 04_fpmc.ipynb               FPMC + persona ablation
│   ├── 05_sasrec.ipynb             SASRec + side-feature ablation
│   └── 06_comparison.ipynb         final table, sliced analyses, plots
├── scripts/download_data.py     one-time data fetch (dev only; NOT part of the pipeline)
├── results.csv                  one row of metrics per model/ablation
├── README.md                    this document
└── PLAN.md                      original design plan
```

**Key functions**
- `preprocessing.preprocess(data_dir, sample_n_trips=None, ...)` → the processed dict (and pickles it).
- `preprocessing.load_processed(dir)` → load the pickle.
- `evaluate.evaluate(ranked_lists, targets)` → the metrics dict.
- `evaluate.evaluate_popularity(data)` → the Popularity baseline / correctness peg.
- `evaluate.evaluate_sliced(ranked_lists, targets, group_ids)` → metrics per subgroup.
- `evaluate.append_result(path, name, metrics)` → append a row to `results.csv`.
- `fpmc.run_fpmc(data, axis="persona"|"user")` → `(metrics, ranked_lists, val_ndcg)`.
- `sasrec.run_sasrec(data, side_feats=..., side_mode="sum"|"concat")` → same shape.

## 10. For the teammate: plug in your own model

Import the two shared files — **don't rewrite them** — and make your model output a
ranked list of city ids per test trip. That's the entire contract.

```python
from preprocessing import load_processed          # or: from src.preprocessing import ...
from evaluate import evaluate, append_result       # or: from src.evaluate import ...

data = load_processed("data/processed")            # same data everyone uses

# Build your model from data["train"] (list of trip dicts; schema in Section 5 output).
#   e.g. order-1 Markov: count transitions cities[i] -> cities[i+1].
#   Remember: ids 0 (PAD) and 1 (UNK) are NOT real cities — never predict them.

ranked_lists = []
for t in data["test"]:
    last_city = t["cities"][-1]                     # observed prefix ends here
    ranked_lists.append(my_ranking(last_city))      # list of city ids, best first, len >= 10

targets = [t["target_city"] for t in data["test"]]
append_result("results.csv", "Markov-order1", evaluate(ranked_lists, targets))
```

Anything that produces a ranked list per test trip (counts, RNN, transformer) slots
straight in and lands in the same `results.csv` for the comparison.

## 11. How to run

### Google Colab (recommended; needed for fast full-scale SASRec)
1. Upload the `project1` folder (or clone the repo).
2. Fetch the data into `data/`:
   ```python
   !mkdir -p data && cd data && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/train_set.csv && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/test_set.csv && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/ground_truth.csv
   ```
3. Run notebooks in order: `01 → 02 → 04 → 05 → 06`.
4. Full-scale SASRec: **Runtime → Change runtime type → GPU** (auto-detected).

### Local
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install pandas numpy matplotlib torch jupyter
python scripts\download_data.py --out data
jupyter notebook                                    # run 01 -> 02 -> 04 -> 05 -> 06
```

**Scale.** Preprocessing defaults to a fast 5k-trip subsample; pass
`sample_n_trips=None` for the full dataset (the numbers in Section 8). FPMC/Popularity
run in minutes on CPU; run full SASRec on a Colab GPU.

## 12. Submission constraints (from the scope)

- The graded code (`src/` + notebooks) reads only from the local `data/` folder —
  **no URLs, no cloud links**. `download_data.py` is a dev convenience, not the pipeline.
- **No datasets and no model checkpoints** are committed (see `.gitignore`).
- Deliverables are the report PDF + contribution PDF + this code zip.
