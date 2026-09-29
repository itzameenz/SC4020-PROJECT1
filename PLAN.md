# Project Plan — Sequential Recommendation on the Booking.com MDT Dataset

**Course:** NTU Technical Review — Recommender Systems / Sequential Prediction
**Deliverables:** a documented codebase + a written report (Abstract / Introduction / Methods / Experiments / Conclusion).

**In one line:** We compare five sequential-recommendation models of increasing
sophistication on the task of predicting a traveller's next city, running them all
through **one shared, leakage-free pipeline** so that every difference in the
results reflects the *model*, not differences in how the data was prepared or
scored.

---

## Table of contents
1. [Goal & research question](#1-goal--research-question)
2. [Dataset](#2-dataset)
3. [Models & ownership](#3-models--ownership)
4. [Design principle: one shared pipeline](#4-design-principle-one-shared-pipeline)
5. [Preprocessing](#5-preprocessing)
6. [Evaluation](#6-evaluation)
7. [The model layer & required ablations](#7-the-model-layer--required-ablations)
8. [Project structure](#8-project-structure)
9. [How to run](#9-how-to-run)
10. [Milestones & division of labour](#10-milestones--division-of-labour)
11. [The story the report tells](#11-the-story-the-report-tells)
12. [Appendix: processed-data schema (the interface)](#appendix-processed-data-schema-the-interface)

---

## 1. Goal & research question

Given the cities a traveller has already booked within a trip, **predict the next
city**. We ask: *as we move from a trivial baseline to a self-attention
transformer, where does each added capability actually pay off — and where does it
not?* The answer, sliced by trip length and destination popularity, is the core
contribution of the report.

The five models each add exactly one capability over the previous:

| Model | Capability it adds |
|---|---|
| Popularity | none (global frequency floor) |
| Markov | memory of the last 1–2 steps |
| FPMC | + personalization |
| GRU4Rec | + long-range sequence memory |
| SASRec | + selective attention over the sequence |

---

## 2. Dataset

**Booking.com Multi-Destination Trips (MDT).** Anonymised hotel reservations
grouped into trips; the rows of a trip, ordered by check-in date, form a sequence
of cities.

- `train_set.csv` — ~1.17 M reservations (full trips).
- `test_set.csv` — trips with the **final city masked** (`city_id = 0`,
  `hotel_country` blank); all earlier rows intact.
- `ground_truth.csv` — the true masked city per trip (`utrip_id, city_id,
  hotel_country`), ~70.7 k trips.

**Columns:** `user_id, checkin, checkout, city_id, device_class, affiliate_id,
booker_country, hotel_country, utrip_id`.

**The split is given to us.** The masked test row + ground truth *is* the
leave-last-out evaluation. We do **not** invent our own test split; we reuse this
directly (verified: the test set's minimum trip length is 4).

---

## 3. Models & ownership

| # | Model | Owner |
|---|---|---|
| 0 | Popularity (sanity floor) | shared — built into the scorer |
| 1 | Markov (order-1 & order-2) | Ronia |
| 2 | **FPMC** | Ameen |
| 3 | GRU4Rec | Ronia |
| 4 | **SASRec** | Ameen |

Everyone imports the **same** `preprocessing.py` and `evaluate.py`. Nobody writes
their own version of either — that is what keeps the five models comparable.

---

## 4. Design principle: one shared pipeline

Two shared files carry the whole comparison:

- **`preprocessing.py`** — raw CSVs → clean sequences + ID mappings + features +
  the train/validation/test split.
- **`evaluate.py`** — scores *any* model identically.

They meet the models through **one interface**: every model, no matter how it works
internally, outputs a **ranked list of city IDs per test trip** (best guess first).
The scorer only looks at where the true city lands in that list. This is why a
frequency count, a Markov transition table and a transformer can all sit in the
same results table.

```
raw CSVs ──preprocessing.py──▶ clean sequences + features + splits
                                        │
              ┌───────────┬─────────────┼─────────────┬───────────┐
          Popularity    Markov        FPMC         GRU4Rec      SASRec
              │           │             │              │           │
              └───────────┴──── ranked list per trip ──┴───────────┘
                                        │
                                 evaluate.py ──▶ HR@k, NDCG@k, MRR, …
```

---

## 5. Preprocessing

All of the following lives in `preprocessing.py` and is applied **identically** to
every model. Steps marked *(added)* extend the team's original agreed list.

### 5.1 Cleaning *(added)*
Conservative, so it can never corrupt a sequence:
- drop exact-duplicate reservation rows,
- drop rows whose check-in date cannot be parsed (they can't be ordered),
- `checkout < checkin` is handled by clipping the length-of-stay feature to ≥ 0
  rather than deleting the row.

### 5.2 Ordering & sequence building
Within each `utrip_id`, sort rows by check-in date to recover the true visit order,
then take the ordered list of cities as the sequence.

### 5.3 Consecutive-duplicate collapse
If the same city appears in two back-to-back rows, collapse it to one — a repeated
stay is a longer stay, not a move to a new destination, and would otherwise teach
every model a trivial self-loop.

### 5.4 Minimum trip length
Drop **training** trips with fewer than 4 stops, to match the test set (whose
trips all have ≥ 4). Length-1 trips are always dropped (nothing to predict). The
**test set is never filtered.** `min_trip_len` is a parameter so this can be
revisited as an ablation.

### 5.5 Categorical encoding
Each categorical is mapped to a contiguous integer using a mapping built **only
from the training set**. Two indices are reserved: **`0 = PAD`** (padding) and
**`1 = UNK`** (any value unseen in training); real values start at `2`.

*Why PAD and UNK must be separate (Ronia's catch):* GRU4Rec and SASRec pad every
sequence up to a fixed length. If padding and "unseen city" shared index `0`, a
padded position would be indistinguishable from a genuine unknown city, and the
models would conflate "nothing here" with "a city we never saw in training".
Splitting them — `0 = PAD`, `1 = UNK` — keeps the two meanings distinct.

Encoded fields: `city_id`, `user_id`, `hotel_country` *(agreed)* — plus
`booker_country`, `device_class`, `affiliate_id` *(added, for FPMC's persona and
SASRec's side-features)*.

### 5.6 Engineered features (a superset — each model takes only its slice)

| Feature | Meaning | Markov | FPMC | GRU4Rec | SASRec |
|---|---|:--:|:--:|:--:|:--:|
| city sequence | the base signal | ✓ | ✓ | ✓ | ✓ |
| **persona** = (booker_country, device_class) | a coarse profile that recurs across trips | — | ✓ | — | opt |
| **trip position** (index / fraction / 5-bucket) | where in the trip a step sits | — | ✓ | — | ✓ (free) |
| **month** of check-in (1–12) | seasonality | — | opt | opt | ✓ |
| **length of stay** (nights) | trip intent | — | opt | opt | ✓ |
| device / country / affiliate | booking context | — | →persona | — | ✓ |

Note on the persona: nearly every `user_id` appears in a **single** trip, so a raw
per-user factor cannot generalise to test (every test user is cold-start). The
persona is a low-cardinality profile that recurs across many trips, giving FPMC
something learnable. FPMC is run with **both** (raw `user_id` vs persona) as a
headline ablation.

Note on position: SASRec learns positional embeddings automatically, so it gets
position "for free"; the explicit feature matters most for **FPMC**, which has no
built-in notion of sequence order.

### 5.7 Splits — three-way and leakage-free *(validation added)*

| Split | Source | Target | Used for |
|---|---|---|---|
| **train** | 90 % of train trips | full sequences | fitting the models |
| **validation** | 10 % of train trips, **last city hidden** | that hidden city | tuning hyperparameters + early stopping |
| **test** | given `test_set` + `ground_truth` | the masked city | **final numbers only** |

The validation set is carved out of train so that FPMC/SASRec can be tuned and
early-stopped **without ever touching the test set** — the single most important
guard against over-optimistic results.

**Tuning protocol *(added — Ronia's point)*:** to keep tuning fair, every trained
model is tuned the **same way** — a small **grid search** over a predefined
hyperparameter grid, with the configuration chosen by **validation NDCG@10** (and
early stopping on the same signal). The count-based models have almost nothing to
tune (Markov: order ∈ {1, 2} and the add-k smoothing constant) and are selected on
validation too. No model ever sees the test set during tuning. We fix this one
protocol across all algorithms so that no model gets an unfair tuning advantage.

### 5.8 Leakage guards
- Encoders are fit on **train only**.
- Validation is carved from **train only**; the test set is read once, at the end.
- Only pre-decision context is ever fed to a model (dates, device, booker country,
  and *past* cities — never anything about the target city itself).

### 5.9 Output
A single pickled dictionary (`data/processed/processed.pkl`) with keys
`train`, `valid`, `test`, `encoders`, `persona_encoder`, `n_cities`, `meta`.
Every model loads this and reads its slice. Full schema in the [Appendix](#appendix-processed-data-schema-the-interface).

**What is deliberately *not* here:** padding/truncating sequences to a fixed length
and building each model's training examples (Markov transition counts, FPMC
(user, prev, next) triples, SASRec shifted sequences) live **inside each model**,
because different models need different formats (e.g. Markov must not see padded
sequences).

---

## 6. Evaluation

All of the following lives in `evaluate.py` and scores every model identically.

### 6.1 Protocol
Leave-last-out, exactly as the dataset ships it: hide the final city of a trip,
have the model rank all cities, and check where the true city lands.

### 6.2 Metrics
For a trip whose true city ends up at rank *r* (1-based) in the model's ranked list:

- **HR@k** (Hit Rate) = 1 if *r* ≤ k, else 0. *"Is the true city in the top k?"*
- **NDCG@k** = 1 / log₂(*r* + 1) if *r* ≤ k, else 0. *"How high up is it?"* (single
  relevant item ⇒ ideal DCG = 1).
- **MRR** (Mean Reciprocal Rank) = 1 / *r* (0 if not found). *"On average, how good
  is its position?"*

All averaged over every test trip, reported for **k ∈ {1, 5, 10}**, ranked against
the **full city catalog**. The reserved indices `0` (PAD) and `1` (UNK) are never
valid predictions; a trip whose true city was unseen in training maps to UNK and
is an automatic miss, as it would be in deployment.

### 6.3 The interface a model must satisfy
```python
from evaluate import evaluate
targets = [t["target_city"] for t in data["test"]]
metrics = evaluate(ranked_lists, targets)     # ranked_lists aligned to data["test"]
```
`ranked_lists[i]` is a list of city indices, best-first, length ≥ 10 (100 is
recommended so MRR is well-defined deep into the list).

### 6.4 Correctness peg
Also compute **Accuracy@4** (= HR@4). The Popularity baseline should reproduce the
repo's demo (~0.053). If it does, the scorer is proven correct — worth a sentence
in the report.

### 6.5 Deeper analyses (turning numbers into findings)

| Analysis | What it reveals |
|---|---|
| Metrics **sliced by trip length** | whether long-range models (GRU4Rec/SASRec) actually win on longer trips — the project's central hypothesis |
| Metrics **sliced by target popularity** (head vs tail) | popularity bias; embedding models should beat Markov on rare destinations |
| **Catalog coverage** | how much of the city catalog a model ever recommends — Popularity collapses to a handful |
| **Multi-seed training** (FPMC, GRU4Rec, SASRec) *(added — Ronia's point)* | run each trained model 2–3 times with different random seeds and report **mean ± spread** on the headline metrics — separates "genuinely better" from a lucky/unlucky training run |
| **Bootstrap confidence intervals** | resample which test trips are scored to see how much a metric wobbles — tells us whether a gap between two models is signal or noise |
| **Efficiency** (#params, train & inference time) | the accuracy-vs-cost trade-off a real comparison must weigh |

The last two capture *different* uncertainties and we report both: **multi-seed**
captures **training** variance (was this a lucky initialisation?), while
**bootstrap** captures **evaluation** variance (did we happen to test on easy
trips?). Only trained models (FPMC, GRU4Rec, SASRec) need multi-seed; the
deterministic count-based models (Popularity, Markov) do not vary across seeds.

### 6.6 Results logging
Every model appends one row to a shared `results.csv`:
`model, HR@1, HR@5, HR@10, NDCG@5, NDCG@10, MRR, Accuracy@4, n_trips`.
The comparison notebook stacks these into the final table and plots.

---

## 7. The model layer & required ablations

Each model derives its own training format from the shared sequences and carries
the ablation the rubric asks for:

| Model | Key ablation / reported analysis |
|---|---|
| **Popularity** | none — reference floor + scorer sanity check |
| **Markov** | order-1 vs order-2; add-k (Laplace) smoothing; **% of test transitions needing fallback** (explains Markov's ceiling) |
| **FPMC** | raw `user_id` **vs** persona; effect of the position feature |
| **GRU4Rec** | with vs without side-features |
| **SASRec** | side-feature ablation (device / country / affiliate; concat vs sum embeddings) |

---

## 8. Project structure

```
project1/
  data/                     raw CSVs (gitignored) + processed/processed.pkl
  notebooks/
    01_data_exploration.ipynb   EDA: trip lengths, sparsity, seasonality
    02_preprocessing.ipynb      runs the shared preprocessing
    03_markov.ipynb             Popularity + Markov            (Ronia)
    04_fpmc.ipynb               FPMC + persona ablation        (Ameen)
    05_gru4rec.ipynb            GRU4Rec                        (Ronia)
    06_sasrec.ipynb             SASRec + side-feature ablation (Ameen)
    07_comparison.ipynb         final table + sliced analyses + plots
  src/
    preprocessing.py          SHARED
    evaluate.py               SHARED
    models/{markov,fpmc,gru4rec,sasrec}.py
  scripts/download_data.py    one-time data fetch (dev only; not the pipeline)
  README.md
  PLAN.md                     (this file)
```

---

## 9. How to run

```bash
pip install pandas numpy matplotlib torch
python scripts/download_data.py --out data   # fetch train/test/ground_truth into data/
```

Then run the notebooks in order: `01 → 02 → (model notebooks) → 06`.

- The pipeline (notebooks + `src/`) only ever reads from the **local `data/`
  folder** — it never references a URL, per the submission rules.
  `download_data.py` is a convenience and is not part of the graded code; datasets
  and checkpoints are not committed.
- Preprocessing uses a **5 k-trip subsample** by default for fast, end-to-end dev
  (`sample_n_trips=5000`); set `sample_n_trips=None` for full-dataset final numbers.
- Preprocessing and the count-based models are CPU-only. **Full-scale SASRec
  training should run on a Colab GPU**, not locally.

---

## 10. Milestones & division of labour

| # | Milestone | Owner | Status |
|---|---|---|---|
| 1 | `preprocessing.py` (+ features, splits, cleaning) | Ameen | built |
| 2 | `evaluate.py` (metrics + sanity peg) | Ronia | in progress |
| 3 | Run 1–2 in Colab; confirm Popularity ≈ 0.053 | both | next |
| 4 | Markov (+ fallback rate), GRU4Rec | Ronia | after 3 |
| 5 | FPMC (+ persona ablation), SASRec (+ side-feature ablation) | Ameen | after 3 |
| 6 | Comparison notebook: table, sliced analyses, plots | both | after 4–5 |
| 7 | Repeat pipeline on a **second dataset** (different domain) | both | required by assignment |
| 8 | Write the report | both | last |

Working environment: one shared Google Colab notebook per person, all importing the
same `preprocessing.py` + `evaluate.py`, with the CSVs loaded into the runtime.

---

## 11. The story the report tells

Not a leaderboard — a diagnosis. Each model adds one capability; our sliced
evaluation shows **where that capability pays off and where it doesn't**:

- Popularity does passably on head destinations but **collapses on coverage** —
  motivating everything that follows.
- Markov captures short-range structure but **plateaus on long trips** and pays a
  visible **fallback tax** on unseen transitions.
- FPMC helps **only where personalization has signal** — the raw-`user_id` vs
  persona gap makes that concrete.
- GRU4Rec and SASRec are expected to win specifically on **longer trips and
  tail destinations**, and the ablations show **which side-features actually
  move the metrics**.

That "where and why", backed by the same metrics on the same split for all five
models, is the Technical Review.

---

## Appendix: processed-data schema (the interface)

`preprocess()` returns / pickles a dict:

```
{
  "train": [ trip, ... ],          # full sequences, for fitting
  "valid": [ trip, ... ],          # last city hidden, for tuning/early-stopping
  "test":  [ trip, ... ],          # given masked test set + ground-truth target
  "encoders": { col: {value: idx} },      # built from train; 0 = PAD, 1 = UNK
  "persona_encoder": { (country, device): idx },
  "n_cities": int,                 # catalog size incl. reserved 0 (PAD) & 1 (UNK)
  "meta": { ... counts, params ... }
}
```

Each `trip` is a dict:

| Field | Type | Meaning |
|---|---|---|
| `utrip_id` | str | trip id |
| `cities` | list[int] | encoded city sequence (for test/valid: the observed **prefix**) |
| `hotel_country` | list[int] | encoded country per step |
| `month` | list[int] | check-in month per step (1–12) |
| `los` | list[int] | length of stay per step (nights) |
| `pos_index` / `pos_frac` / `pos_bucket` | list | trip position per step |
| `user` | int | encoded user_id |
| `persona` | int | encoded (booker_country, device_class) |
| `booker_country` / `device` / `affiliate` | int | encoded trip-level context |
| `query` | dict/None | context of the step being predicted (month, los); `None` for train |
| `target_city` | int | **(valid/test only)** the true next city to predict |

**Model contract:** consume a trip's `cities` (+ whatever features your model
uses), output a ranked list of city indices per **test** trip, aligned to
`data["test"]`, then call `evaluate(ranked_lists, [t["target_city"] for t in
data["test"]])`.