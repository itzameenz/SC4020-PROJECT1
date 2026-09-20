# SC4020 Project 1 — Sequential Recommendation on Booking.com MDT

Predict a traveller's **next city** in a multi-city trip, and compare five models
of increasing sophistication — all run through **one shared, leakage-free
pipeline** so the comparison is fair. This README is written so any teammate can
read it cold, understand every file, and plug in their own model.

Full design rationale: [`PLAN.md`](PLAN.md).

---

## 1. Results at a glance (full dataset: 190,737 train / 21,187 val / 70,662 test)

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 | MRR | Accuracy@4 |
|---|---|---|---|---|---|---|
| Popularity (floor) | 0.018 | 0.052 | 0.102 | 0.050 | 0.035 | 0.043 |
| FPMC-persona | 0.192 | 0.432 | 0.531 | **0.350** | 0.294 | 0.397 |
| FPMC-user | 0.192 | 0.428 | 0.515 | 0.345 | 0.291 | 0.395 |
| SASRec (no side) | 0.183 | 0.434 | **0.549** | **0.351** | 0.289 | 0.396 |
| SASRec (side-sum) | 0.158 | 0.416 | 0.537 | 0.332 | 0.268 | 0.378 |
| SASRec (side-concat) | 0.131 | 0.360 | 0.494 | 0.290 | 0.228 | 0.316 |

*(numbers live in [`results.csv`](results.csv); Markov + GRU4Rec rows are added by the teammate)*

**Headline findings**
- Every real model beats the Popularity floor by ~7× on NDCG@10.
- **FPMC-persona ≥ FPMC-user** on full data — with ~200k mostly-single-trip users a raw per-user factor is cold at test, while the coarse persona recurs and generalises. (The ordering flips on a small subsample — worth noting.)
- **SASRec's side features *hurt*** (no-side > sum > concat) — the trip-level context is too coarse/noisy here; a clean, honest ablation result.
- SASRec (no-side) and FPMC-persona are neck-and-neck (best HR@10 vs best NDCG@10).

---

## 2. The task and the dataset

Each **trip** (`utrip_id`) is an ordered sequence of cities a traveller booked.
Given the cities so far, predict the next one — classic **sequential
recommendation**.

Three CSVs (from <https://github.com/bookingcom/ml-dataset-mdt>, placed in `data/`):

| File | What it is |
|---|---|
| `train_set.csv` | ~1.17M reservations, full trips — used to train |
| `test_set.csv` | same format, but each trip's **final city is masked** (`city_id=0`) |
| `ground_truth.csv` | the true masked city per test trip — used only for scoring |

**Columns:** `utrip_id, user_id, checkin, checkout, city_id, hotel_country,
booker_country, device_class, affiliate_id`. The item we predict is `city_id`.

The masked test set + ground truth **is** the leave-last-out split — we reuse it
directly, never inventing our own. (Assignment needs ≥2 datasets; MDT is #1, a
second-domain dataset is still to be added.)

---

## 3. Repository walkthrough (what every file does)

```
project1/
├── data/                        raw CSVs (gitignored) + processed/processed.pkl
├── src/
│   ├── preprocessing.py         SHARED — raw CSVs -> clean sequences + split
│   ├── evaluate.py              SHARED — scores any model (HR/NDCG/MRR ...)
│   └── models/
│       ├── fpmc.py              our model #2 (FPMC)
│       └── sasrec.py            our model #4 (SASRec)
├── notebooks/
│   ├── 01_data_exploration.ipynb   EDA (trip lengths, sparsity, seasonality)
│   ├── 02_preprocessing.ipynb      runs preprocessing, writes processed.pkl
│   ├── 04_fpmc.ipynb               FPMC + persona ablation
│   ├── 05_sasrec.ipynb             SASRec + side-feature ablation
│   └── 06_comparison.ipynb         final table + sliced analyses + plots
├── scripts/download_data.py     one-time data fetch (dev only; NOT the pipeline)
├── results.csv                  one row of metrics per model
├── README.md                    this file
└── PLAN.md                      full design rationale
```

### The two shared files (everyone imports these — do **not** rewrite them)

**`src/preprocessing.py`** turns the raw CSVs into clean, encoded sequences that
every model consumes. It:
1. cleans (drops duplicate / unparseable rows),
2. orders each trip by check-in date,
3. collapses consecutive repeats of the same city,
4. drops train trips with < 4 stops (matches the test set; test is never filtered),
5. encodes all categoricals **from train only** — `0 = PAD`, `1 = UNK`, real ids from 2,
6. adds features (month, length-of-stay, trip position, and a `persona`),
7. splits train into **train (90%)** + **validation (10%, last city hidden)**, and
   attaches the ground-truth target to each **test** trip.

Output: one dict (pickled to `data/processed/processed.pkl`) — see the schema in §5.

**`src/evaluate.py`** scores any model’s predictions:
- `evaluate(ranked_lists, targets)` → `HR@{1,5,10}`, `NDCG@{5,10}`, `MRR`, `Accuracy@4`.
- `evaluate_popularity(data)` → the Popularity baseline (and correctness peg).
- `evaluate_sliced(...)`, `city_frequency(data)` → the by-length / by-popularity analyses.
- `append_result(path, model_name, metrics)` → adds one row to `results.csv`.

### Our two models
- **`src/models/fpmc.py`** — FPMC (matrix factorization + factorized Markov chain,
  BPR loss). `run_fpmc(data, axis="persona"|"user")` trains, predicts, and scores.
- **`src/models/sasrec.py`** — SASRec (causal self-attention transformer, BPR).
  `run_sasrec(data, side_feats=..., side_mode="sum"|"concat")`.

---

## 4. For the teammate — plug in your own model (Markov / GRU4Rec)

You only need the two shared files. **Import them, don’t rewrite them** — that’s
what keeps all five models comparable. Your model just has to output, for each
test trip, a **ranked list of city ids** (best guess first, length ≥ 10).

```python
from preprocessing import preprocess, load_processed   # or: from src.preprocessing import ...
from evaluate import evaluate, append_result            # or: from src.evaluate import ...

# 1. Load the SAME processed data everyone uses
data = load_processed("data/processed")   # or preprocess(data_dir="data") the first time

# 2. Build your model from data["train"]  (a list of trip dicts, see schema below)
#    e.g. order-1 Markov: count transitions cities[i] -> cities[i+1]
#    remember: ids 0 (PAD) and 1 (UNK) are NOT real cities — never predict them.

# 3. Predict: for each TEST trip, output a ranked list of city ids
ranked_lists = []
for t in data["test"]:
    last_city = t["cities"][-1]              # the observed prefix ends here
    ranked_lists.append(my_ranking(last_city))   # list of city ids, best first, len>=10

# 4. Score with the shared evaluator and log it
targets = [t["target_city"] for t in data["test"]]
metrics = evaluate(ranked_lists, targets)
append_result("results.csv", "Markov-order1", metrics)
print(metrics)
```

That’s the whole contract. Anything that produces a ranked list per test trip
(counts, RNN, transformer) slots straight in and lands in the same `results.csv`.

---

## 5. The processed-data schema (the interface)

`preprocess()` / `load_processed()` return a dict:

```
{ "train": [trip, ...],   # full sequences, for fitting
  "valid": [trip, ...],   # last city hidden, for tuning / early stopping
  "test":  [trip, ...],   # observed prefix + ground-truth target
  "encoders": {...}, "persona_encoder": {...},
  "n_cities": int,        # catalog size incl. PAD(0) and UNK(1)
  "meta": {...} }
```

Each `trip` is a dict:

| Field | Meaning |
|---|---|
| `cities` | encoded city sequence (for valid/test: the observed **prefix**) |
| `hotel_country`, `month`, `los` | per-step features |
| `pos_index` / `pos_frac` / `pos_bucket` | trip position per step |
| `user`, `persona`, `booker_country`, `device`, `affiliate` | trip-level context |
| `query` | context (month, los) of the step being predicted |
| `target_city` | **(valid/test only)** the true next city — what you predict |

---

## 6. How to run

### Google Colab (recommended; required for fast full-scale SASRec)
1. Upload the `project1` folder (or clone the repo).
2. Fetch the data into `data/`:
   ```python
   !mkdir -p data && cd data && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/train_set.csv && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/test_set.csv && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/ground_truth.csv
   ```
3. Run notebooks in order: `01 → 02 → 04 → 05 → 06`.
4. For full-scale SASRec: **Runtime → Change runtime type → GPU** (auto-detected).

### Local
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install pandas numpy matplotlib torch jupyter
python scripts\download_data.py --out data
jupyter notebook                                     # run 01 -> 02 -> 04 -> 05 -> 06
```

**Scale:** preprocessing takes `sample_n_trips=5000` for a fast dev run, or
`sample_n_trips=None` for the full dataset (the numbers in §1). FPMC/Popularity run
in minutes on CPU; **full SASRec should use a Colab GPU** (CPU takes ~15–20 min per
config).

---

## 7. Analysis notes for the report

- **Sanity peg:** computed the repo’s way (raw city counts), Popularity’s
  Accuracy@4 = 0.0527, reproducing the repo’s ~0.053 → the scorer is correct.
- **Parameter setting to discuss:** FPMC’s validation NDCG peaks at epoch 1 then
  declines — it converges/overfits fast; lowering the learning rate and adding
  regularization is the natural fix (good "parameter settings" content).
- **Ablations:** FPMC persona-vs-user, and SASRec side-features (none/sum/concat).
- **Sliced analysis (`06`):** metrics by trip length and by destination
  popularity — models are strong on head cities and weak on the tail (cold-tail
  problem).

## 8. Submission constraints (from the scope)
- The graded code (`src/` + notebooks) reads only from the local `data/` folder —
  **no URLs, no cloud links**. `download_data.py` is dev-only.
- **No datasets, no model checkpoints** are committed (see `.gitignore`).
