# Sequential Recommendation on the Booking.com MDT Dataset

A Technical Review project (NTU) comparing five sequential-recommendation models
on the task of **predicting a traveller's next city**, all scored through one
shared, leakage-free pipeline so the comparison is fair. This README is the
practical guide; the full design rationale lives in [`PLAN.md`](PLAN.md).

---

## 1. What the dataset is

**Booking.com Multi-Destination Trips (MDT).** Each row is one anonymised hotel
reservation; reservations are grouped into **trips** (`utrip_id`), and the rows of
a trip — ordered by check-in date — form a **sequence of cities** a traveller
visited. Predicting the next city in that sequence is a classic *sequential
recommendation* problem.

**Columns**

| Column | Meaning |
|---|---|
| `utrip_id` | trip id — groups the reservations of one trip |
| `user_id` | traveller id (most users appear in only one trip) |
| `checkin`, `checkout` | stay dates (define order + length of stay) |
| `city_id` | the city booked — **this is the item we predict** |
| `hotel_country` | country of the city |
| `booker_country` | country the booking was made from |
| `device_class` | desktop / mobile / tablet |
| `affiliate_id` | booking channel |

Source: <https://github.com/bookingcom/ml-dataset-mdt>

## 2. How the data is split (train / test / ground truth)

The split is **provided by the dataset** — we do not invent our own:

- **`train_set.csv`** — ~1.17M reservations, full trips. Used to train.
- **`test_set.csv`** — the same format, but each trip's **final city is masked**
  (`city_id = 0`, `hotel_country` blank). The model sees the trip *up to* the last
  stop and must predict that last city.
- **`ground_truth.csv`** (`utrip_id, city_id, hotel_country`, ~70.7k trips) — the
  true masked city, used only for scoring.

This is exactly the **leave-last-out** protocol the assignment asks for: hide the
last step of each sequence, predict it, compare against the truth.

**Two-dataset requirement:** the assignment needs ≥2 datasets. MDT is dataset #1;
once this pipeline is solid we run the *same five models* on a second dataset from
a different domain (TBD) for cross-domain comparison.

## 3. The models

| # | Model | What it adds | Owner |
|---|---|---|---|
| 0 | Popularity | nothing — a frequency floor | shared (in `evaluate.py`) |
| 1 | Markov (order-1/2) | memory of the last 1–2 steps | teammate |
| 2 | **FPMC** | + personalization | **us** (`src/models/fpmc.py`) |
| 3 | GRU4Rec | + long-range memory | teammate |
| 4 | **SASRec** | + self-attention | **us** (`src/models/sasrec.py`) |

Everyone imports the **same** `src/preprocessing.py` and `src/evaluate.py`.

## 4. Preprocessing (`src/preprocessing.py`)

Turns the raw CSVs into clean, encoded sequences shared by every model:

1. **Clean** — drop duplicate/unparseable rows.
2. **Order** each trip by check-in date.
3. **Collapse** consecutive repeats of the same city.
4. **Filter** train trips to ≥4 stops (matches the test set); test is never filtered.
5. **Encode** all categoricals from **train only**; `0 = PAD`, `1 = UNK`, real ids from 2.
6. **Features** — month, length of stay, trip position (index/fraction/bucket), and a
   persona = (booker_country, device_class) for FPMC.
7. **Split** train into **train (90%)** and **validation (10%, last city hidden)** so
   the neural models can be tuned without ever touching the test set.

Output: one pickle `data/processed/processed.pkl` with `train`, `valid`, `test`,
`encoders`, `n_cities`, `meta`.

## 5. Evaluation (`src/evaluate.py`)

- **Interface:** every model outputs a *ranked list of city ids per test trip*.
- **Metrics:** **HR@k, NDCG@k, MRR for k ∈ {1, 5, 10}**, ranked against the full
  catalog; plus **Accuracy@4** as a sanity peg (Popularity reproduces the repo's
  ~0.053 → proves the scorer is correct).
- **Sliced analysis:** `evaluate_sliced` reports metrics by trip length and by
  destination popularity (head/tail) for the deeper findings.

## 6. Our two algorithms

- **FPMC** (`src/models/fpmc.py`) — matrix factorization (user/persona × city) fused
  with a factorized Markov transition (city × city), trained with **BPR**. Ablation:
  raw `user_id` vs **persona**; plus a position-bucket feature.
- **SASRec** (`src/models/sasrec.py`) — a causal self-attention transformer over the
  city sequence (hand-rolled attention for full control of the masks), next-item BPR
  objective. Ablation: side features **off / summed / concatenated**
  (device, booker_country, affiliate).

## 7. Project structure

```
project1/
  data/                     raw CSVs (gitignored) + processed/processed.pkl
  notebooks/
    01_data_exploration.ipynb   EDA
    02_preprocessing.ipynb      runs the shared preprocessing
    04_fpmc.ipynb               FPMC + persona ablation      (ours)
    05_sasrec.ipynb             SASRec + side-feature ablation (ours)
    06_comparison.ipynb         table + sliced analyses + plots
  src/
    preprocessing.py          shared
    evaluate.py               shared
    models/fpmc.py, sasrec.py
  scripts/download_data.py    one-time data fetch (dev only; not the pipeline)
  PLAN.md, README.md
```

---

## 8. How to run — step by step

### Option A — Google Colab (recommended, and needed for full-scale SASRec)

1. Upload the `project1` folder to your Google Drive, or upload the files to the
   Colab session.
2. Open a notebook and, in the first code cell area, get the data into `data/`:
   ```python
   !mkdir -p data && cd data && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/train_set.csv && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/test_set.csv && \
     wget -q https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/ground_truth.csv
   ```
3. Run the notebooks **in order**: `01 → 02 → 04 → 05 → 06`.
4. For full-scale SASRec: **Runtime → Change runtime type → GPU**, and it will train
   on the GPU automatically (`device='cuda'` is auto-detected).

### Option B — Local (Windows)

From the `project1` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib torch jupyter
python scripts\download_data.py --out data
jupyter notebook           # then run 01 -> 02 -> 04 -> 05 -> 06
```

> **Import troubleshooting:** if a notebook can't find `src`, the first cell already
> adds the project root to `sys.path`. If Python itself can't find installed
> packages (a mis-configured interpreter), run with an explicit path, e.g.
> `PYTHONPATH="<project1>;<site-packages>" python ...`.

### What each stage produces
- `02` → `data/processed/processed.pkl` (the shared data).
- `04`, `05` → append rows to `results.csv`.
- `06` → the comparison table, bar chart, and sliced analyses.

### Scale
Preprocessing defaults to a **5,000-trip subsample** (`sample_n_trips=5000`) so
everything runs end-to-end in minutes on CPU. For final report numbers, call
`preprocess(..., sample_n_trips=None)` and train SASRec on a Colab GPU.

---

## 9. Illustrative results (5k subsample, CPU — dev numbers, not final)

| Model | HR@10 | NDCG@10 | MRR | Accuracy@4 |
|---|---|---|---|---|
| Popularity | 0.099 | 0.046 | 0.030 | 0.043 |
| FPMC-persona | 0.317 | 0.216 | 0.184 | 0.240 |
| FPMC-user | 0.351 | 0.235 | 0.198 | 0.260 |
| SASRec (no side) | **0.379** | 0.220 | 0.171 | 0.250 |

All three real models beat the Popularity floor by ~4–5× on NDCG@10. On this small
sample the last-city transition dominates, so FPMC and SASRec are close; SASRec's
attention advantage is expected to widen on the full dataset.

**Sanity peg:** on the *full* training set, Popularity's Accuracy@4 = **0.0527**,
reproducing the repo's ~0.053 — confirming the scorer is correct. (On the 5k
subsample it is lower, 0.043, because the frequency ranking is estimated from fewer
trips.)

**Sliced finding (from `06`):** by destination popularity, every model scores
~0.35–0.37 NDCG@10 on **head** cities but ≈0 on **tail** cities — a clear cold-tail
problem worth discussing. By trip length the metrics are roughly flat on this
sample; the long-range story should be revisited on the full data.

Final numbers go in the report after the full-data + GPU run.

Notes / checkpoints and datasets are **not** committed (see `.gitignore`); the
pipeline only reads from the local `data/` folder — it never references a URL.
