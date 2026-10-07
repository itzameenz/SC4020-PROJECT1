"""
preprocessing.py
================
Shared preprocessing for the Booking.com Multi-Destination Trips (MDT) project.

ALL models in this project -- Popularity, Markov (order-1/2), FPMC, GRU4Rec and
SASRec -- import and use THIS module. Identical sequences, identical id mappings
and identical filtering are what make the final comparison table apples-to-apples.
If two teammates preprocess differently, their model scores are not comparable,
even if both models are implemented correctly.

Pipeline (each step is explained in report-ready prose in the notebooks):

1.  Clean. Drop exact-duplicate rows and rows whose check-in date cannot be parsed
    (they cannot be ordered). Conservative -- it never corrupts a sequence.
2.  Order. Sort each trip's rows by check-in date to recover the visit order.
3.  Collapse consecutive-duplicate cities (a repeated stay is one longer stay, not
    a move to a new destination).
4.  Filter. Drop TRAIN trips shorter than MIN_TRIP_LEN (default 4) to match the
    test set (all test trips have >= 4 stops). The test set is never filtered.
5.  Encode. Map every categorical to a contiguous integer using a mapping built
    ONLY from TRAIN. Two indices are reserved:
        0 = PAD  (padding, added later by the neural models' data loaders)
        1 = UNK  (a value unseen in training)
    Real values start at 2. Keeping PAD and UNK separate matters for GRU4Rec /
    SASRec: a padded position must not look like a genuine unknown city.
6.  Features. Per step: month of check-in (1-12), length of stay (nights), and
    trip position (index, 0-1 fraction, and a 5-bucket version). Plus a per-trip
    "persona" = (booker_country, device_class) for the FPMC persona ablation.
7.  Split three ways, leakage-free:
        train      -- (1 - val_frac) of train trips, full sequences
        validation -- val_frac of train trips with the last city hidden
        test       -- the given test_set (masked final row) + ground_truth
    Validation lets FPMC/SASRec be tuned and early-stopped without ever touching
    test.

For the FINAL fit (benchmark protocol step 3) `merge_train_val(data)` folds the
validation trips back into train: it restores each leave-last-out trip to its
full sequence, so the final models train on train + validation.

Output: one dict (also pickled) with keys train / valid / test / encoders /
persona_encoder / n_cities / meta. Every model reads its slice from this.
Padding and per-model training-example construction live inside each MODEL, not
here, because different models need different formats.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Categorical columns that get integer-encoded (mappings built from TRAIN only).
CATEGORICALS: List[str] = [
    "city_id",
    "user_id",
    "hotel_country",
    "booker_country",
    "device_class",
    "affiliate_id",
]

PAD = 0            # padding position (emitted only by model data loaders)
UNK = 1            # a categorical value unseen in the training set
N_RESERVED = 2     # indices 0 and 1 are reserved; real ids start at 2
N_POS_BUCKETS = 5  # number of buckets for the FPMC position feature

RAW_FILES = {
    "train": "train_set.csv",
    "test": "test_set.csv",
    "ground_truth": "ground_truth.csv",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(data_dir: str):
    """Load the three raw CSVs from a LOCAL directory (no cloud links here).

    Populate `data_dir` beforehand with scripts/download_data.py or by copying
    the CSVs in manually. Returns (train_df, test_df, ground_truth_df).
    """
    def _read(name):
        path = os.path.join(data_dir, RAW_FILES[name])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/download_data.py or place the CSV "
                f"in {data_dir} first."
            )
        return pd.read_csv(path)

    return _read("train"), _read("test"), _read("ground_truth")


# ---------------------------------------------------------------------------
# Per-row preparation (cleaning + dates + engineered scalar features)
# ---------------------------------------------------------------------------

def prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, parse dates, derive month + length-of-stay, sort by trip/check-in.

    Cleaning is conservative so it never corrupts a sequence: drop exact-duplicate
    rows and drop rows whose check-in cannot be parsed. A masked test row keeps a
    valid check-in (only its city is hidden), so it survives. checkout < checkin is
    handled by clipping length-of-stay to >= 0 rather than dropping the row.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df["checkin"] = pd.to_datetime(df["checkin"], errors="coerce")
    df["checkout"] = pd.to_datetime(df["checkout"], errors="coerce")
    df = df[df["checkin"].notna()]                        # keep only orderable rows
    df = df.sort_values(["utrip_id", "checkin"]).reset_index(drop=True)
    df["month"] = df["checkin"].dt.month.fillna(0).astype(int)         # 1-12
    los = (df["checkout"] - df["checkin"]).dt.days
    df["length_of_stay"] = los.fillna(0).clip(lower=0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Categorical encoders (built from TRAIN, applied everywhere)
# ---------------------------------------------------------------------------

def build_encoders(train_df: pd.DataFrame,
                   cols: List[str] = CATEGORICALS) -> Dict[str, Dict]:
    """Map each categorical value -> contiguous int starting at 2 (0=PAD, 1=UNK).

    Values are sorted deterministically so the mapping is reproducible across
    machines and teammates.
    """
    encoders: Dict[str, Dict] = {}
    for col in cols:
        values = sorted(train_df[col].dropna().unique().tolist(), key=lambda v: str(v))
        encoders[col] = {v: i + N_RESERVED for i, v in enumerate(values)}
    return encoders


def build_persona_encoder(train_df: pd.DataFrame) -> Dict[tuple, int]:
    """Persona = (booker_country, device_class). Built from TRAIN only."""
    pairs = (
        train_df[["booker_country", "device_class"]].dropna()
        .apply(lambda r: (r["booker_country"], r["device_class"]), axis=1)
        .unique().tolist()
    )
    pairs = sorted(pairs, key=lambda v: str(v))
    return {p: i + N_RESERVED for i, p in enumerate(pairs)}


def _encode(value, mapping: Dict) -> int:
    """Look up one value; missing/unseen -> UNK (1)."""
    if pd.isna(value):
        return UNK
    return mapping.get(value, UNK)


# ---------------------------------------------------------------------------
# Position features
# ---------------------------------------------------------------------------

def _position_features(length: int):
    """Return (index, fraction, bucket) arrays for a sequence of the given length."""
    idx = np.arange(length, dtype=int)
    frac = idx / max(length - 1, 1)                          # 0.0 .. 1.0
    bucket = np.minimum((frac * N_POS_BUCKETS).astype(int), N_POS_BUCKETS - 1)
    return idx, frac, bucket


# ---------------------------------------------------------------------------
# Trip building
# ---------------------------------------------------------------------------

def _collapse_consecutive(records: List[dict]) -> List[dict]:
    """Drop a step whose city equals the immediately preceding step's city."""
    out: List[dict] = []
    prev_city = None
    for rec in records:
        if rec["city"] == prev_city:
            continue
        out.append(rec)
        prev_city = rec["city"]
    return out


def _finalise(steps: List[dict], first_row, persona_encoder, encoders,
              utrip_id) -> dict:
    """Assemble a per-trip record from ordered, de-duplicated step dicts."""
    length = len(steps)
    idx, frac, bucket = _position_features(length)
    persona = persona_encoder.get(
        (first_row["booker_country"], first_row["device_class"]), UNK
    )
    return {
        "utrip_id": utrip_id,
        "cities": [s["city"] for s in steps],
        "hotel_country": [s["hotel_country"] for s in steps],
        "month": [s["month"] for s in steps],
        "los": [s["los"] for s in steps],
        "pos_index": idx.tolist(),
        "pos_frac": [round(f, 4) for f in frac.tolist()],
        "pos_bucket": bucket.tolist(),
        "user": _encode(first_row["user_id"], encoders["user_id"]),
        "persona": persona,
        "booker_country": _encode(first_row["booker_country"], encoders["booker_country"]),
        "device": _encode(first_row["device_class"], encoders["device_class"]),
        "affiliate": _encode(first_row["affiliate_id"], encoders["affiliate_id"]),
        "query": None,
    }


def _build_trip(group: pd.DataFrame, encoders, persona_encoder, is_test: bool):
    """Turn one utrip_id's rows into a per-trip record of encoded sequences.

    For the test set the final row is masked (city_id == 0): its context (month,
    length-of-stay) is kept as the `query` for the step being predicted, but it is
    not added to the observed city sequence.
    """
    city_map, hc_map = encoders["city_id"], encoders["hotel_country"]
    steps, query = [], None
    for _, row in group.iterrows():
        if is_test and row["city_id"] == 0:                  # masked final stop
            query = {"month": int(row["month"]), "los": int(row["length_of_stay"])}
            continue
        steps.append({
            "city": _encode(row["city_id"], city_map),
            "hotel_country": _encode(row["hotel_country"], hc_map),
            "month": int(row["month"]),
            "los": int(row["length_of_stay"]),
        })
    steps = _collapse_consecutive(steps)
    trip = _finalise(steps, group.iloc[0], persona_encoder, encoders,
                     group.iloc[0]["utrip_id"])
    trip["query"] = query
    return trip


def build_trips(df: pd.DataFrame, encoders, persona_encoder, is_test: bool):
    """Build per-trip records for every utrip_id in df (order preserved)."""
    return [_build_trip(g, encoders, persona_encoder, is_test)
            for _, g in df.groupby("utrip_id", sort=False)]


def _to_leave_last_out(trip: dict) -> Optional[dict]:
    """Turn a full TRAIN trip into a validation example: hide its last city.

    The observed prefix is all-but-last; the target is the last city; the last
    step's context becomes the `query`. Returns None if the trip is too short.
    """
    if len(trip["cities"]) < 2:
        return None
    n = len(trip["cities"]) - 1
    idx, frac, bucket = _position_features(n)
    out = dict(trip)
    out["target_city"] = trip["cities"][-1]
    out["cities"] = trip["cities"][:-1]
    out["hotel_country"] = trip["hotel_country"][:-1]
    out["month"] = trip["month"][:-1]
    out["los"] = trip["los"][:-1]
    out["pos_index"] = idx.tolist()
    out["pos_frac"] = [round(f, 4) for f in frac.tolist()]
    out["pos_bucket"] = bucket.tolist()
    out["query"] = {"month": trip["month"][-1], "los": trip["los"][-1]}
    return out


def _restore_full_trip(v: dict) -> dict:
    """Undo `_to_leave_last_out`: put the hidden last city back on a validation trip.

    The target city is appended to `cities`; the last step's month / length of
    stay come back from `query`. `hotel_country` of that last step was not kept
    by the leave-last-out split, so it is filled with UNK (no model that merges
    train+val -- Markov, GRU4Rec -- reads it). Position features are recomputed.
    """
    out = dict(v)
    n = len(v["cities"]) + 1
    idx, frac, bucket = _position_features(n)
    out["cities"] = v["cities"] + [v["target_city"]]
    out["hotel_country"] = v["hotel_country"] + [UNK]
    out["month"] = v["month"] + [v["query"]["month"]]
    out["los"] = v["los"] + [v["query"]["los"]]
    out["pos_index"] = idx.tolist()
    out["pos_frac"] = [round(f, 4) for f in frac.tolist()]
    out["pos_bucket"] = bucket.tolist()
    out["query"] = None
    out.pop("target_city", None)
    return out


def merge_train_val(data: dict) -> dict:
    """Return a copy of `data` whose "train" is train + validation (full trips).

    Used for the FINAL fit of every model (benchmark protocol step 3): after
    hyper-parameters and the epoch count are chosen on validation, the models are
    refit on train + validation and scored once on test.

    * Each validation trip (prefix + hidden last city) is restored to a full
      trip identical in format to a train trip.
    * "valid" is set to [] in the returned dict so nothing can accidentally be
      scored on data the model has trained on. "test" is untouched.
    * The input dict is not modified.
    """
    merged = dict(data)
    restored = [_restore_full_trip(v) for v in data["valid"]]
    merged["train"] = list(data["train"]) + restored
    merged["valid"] = []
    merged["meta"] = {**data["meta"],
                      "merged_train_val": True,
                      "n_train_trips": len(merged["train"]),
                      "n_valid_trips": 0}
    return merged


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def preprocess(
    data_dir: str = "data",
    out_dir: str = "data/processed",
    min_trip_len: int = 4,
    val_frac: float = 0.1,
    sample_n_trips: Optional[int] = None,
    seed: int = 42,
    save: bool = True,
) -> dict:
    """Run the full shared pipeline and (optionally) pickle the result.

    Parameters
    ----------
    data_dir       : folder holding the raw CSVs.
    out_dir        : folder to write processed.pkl into.
    min_trip_len   : trips shorter than this are dropped from TRAIN (default 4).
    val_frac       : fraction of train trips held out for validation (last city
                     hidden). Set 0.0 to disable.
    sample_n_trips : if set, keep only this many random TRAIN trips (fast dev).
                     Set None for the full dataset (final report numbers).
    seed           : RNG seed for the subsample and the train/val split.
    save           : write data/processed/processed.pkl.
    """
    rng = np.random.default_rng(seed)

    train_df, test_df, gt_df = load_raw(data_dir)
    train_df = prepare_rows(train_df)
    test_df = prepare_rows(test_df)

    # Encoders are built from TRAIN only -- the single source of truth for ids.
    encoders = build_encoders(train_df)
    persona_encoder = build_persona_encoder(train_df)

    # --- filter short trips, optional subsample -----------------------------
    lengths = train_df.groupby("utrip_id")["city_id"].size()
    keep = lengths[lengths >= min_trip_len].index
    train_df = train_df[train_df["utrip_id"].isin(keep)]

    if sample_n_trips is not None:
        ids = train_df["utrip_id"].unique()
        chosen = rng.choice(ids, size=min(sample_n_trips, len(ids)), replace=False)
        train_df = train_df[train_df["utrip_id"].isin(set(chosen))]

    # --- split remaining train trips into train / validation ----------------
    ids = rng.permutation(train_df["utrip_id"].unique())
    n_val = int(len(ids) * val_frac)
    val_ids, tr_ids = set(ids[:n_val].tolist()), set(ids[n_val:].tolist())

    train_trips = build_trips(train_df[train_df["utrip_id"].isin(tr_ids)],
                              encoders, persona_encoder, is_test=False)
    train_trips = [t for t in train_trips if len(t["cities"]) >= 2]

    val_full = build_trips(train_df[train_df["utrip_id"].isin(val_ids)],
                           encoders, persona_encoder, is_test=False)
    valid_trips = [v for v in (_to_leave_last_out(t) for t in val_full)
                   if v is not None]

    # --- TEST: attach ground-truth target, never filtered -------------------
    gt_map = dict(zip(gt_df["utrip_id"], gt_df["city_id"]))
    test_trips = build_trips(test_df, encoders, persona_encoder, is_test=True)
    for t in test_trips:
        t["target_city"] = _encode(gt_map.get(t["utrip_id"]), encoders["city_id"])
        t["target_city_raw"] = gt_map.get(t["utrip_id"])

    n_cities = len(encoders["city_id"]) + N_RESERVED   # incl. PAD (0) and UNK (1)

    result = {
        "train": train_trips,
        "valid": valid_trips,
        "test": test_trips,
        "encoders": encoders,
        "persona_encoder": persona_encoder,
        "n_cities": n_cities,
        "meta": {
            "min_trip_len": min_trip_len,
            "val_frac": val_frac,
            "sample_n_trips": sample_n_trips,
            "seed": seed,
            "n_train_trips": len(train_trips),
            "n_valid_trips": len(valid_trips),
            "n_test_trips": len(test_trips),
            "n_cities": n_cities,
            "n_users": len(encoders["user_id"]),
            "n_personas": len(persona_encoder),
            "n_test_targets_unseen": sum(1 for t in test_trips
                                         if t["target_city"] == UNK),
        },
    }

    if save:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "processed.pkl"), "wb") as f:
            pickle.dump(result, f)

    return result


def load_processed(out_dir: str = "data/processed") -> dict:
    """Load a previously pickled preprocessing result."""
    with open(os.path.join(out_dir, "processed.pkl"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Smoke test on synthetic data:  python -m src.preprocessing
# ---------------------------------------------------------------------------

def _synthetic_frames():
    rows = []

    def add(utrip, cities, user, bc, dev, hc):
        for i, c in enumerate(cities):
            rows.append(dict(user_id=user, checkin=f"2016-0{1+i}-01",
                             checkout=f"2016-0{1+i}-03", city_id=c,
                             device_class=dev, affiliate_id=100, booker_country=bc,
                             hotel_country=hc, utrip_id=utrip))

    add("A_1", [10, 20, 30, 40], "u1", "X", "desktop", "CtryA")
    add("B_1", [11, 11, 21, 31, 41], "u2", "Y", "mobile", "CtryB")  # has a dup
    add("C_1", [12, 22], "u3", "X", "desktop", "CtryA")             # too short
    train = pd.DataFrame(rows)

    trows = []
    for i, c in enumerate([10, 20, 30, 0]):                          # 0 = masked
        trows.append(dict(user_id="u1", checkin=f"2017-0{1+i}-01",
                          checkout=f"2017-0{1+i}-02", city_id=c,
                          device_class="desktop", affiliate_id=100,
                          booker_country="X",
                          hotel_country=("CtryA" if c else np.nan), utrip_id="T_1"))
    test = pd.DataFrame(trows)
    gt = pd.DataFrame([dict(utrip_id="T_1", city_id=40, hotel_country="CtryA")])
    return train, test, gt


if __name__ == "__main__":
    import tempfile

    tmp = tempfile.mkdtemp()
    tr, te, gt = _synthetic_frames()
    tr.to_csv(os.path.join(tmp, RAW_FILES["train"]), index=False)
    te.to_csv(os.path.join(tmp, RAW_FILES["test"]), index=False)
    gt.to_csv(os.path.join(tmp, RAW_FILES["ground_truth"]), index=False)

    out = preprocess(data_dir=tmp, out_dir=tmp, min_trip_len=4, val_frac=0.0,
                     save=False)
    print("=== smoke test ===")
    print("n_cities:", out["n_cities"], "(incl PAD=0, UNK=1)")
    print("meta:", out["meta"])
    kept = sorted(t["utrip_id"] for t in out["train"])
    print("train trips kept:", kept)
    assert kept == ["A_1", "B_1"], "C_1 (len 2) should be dropped"
    b = next(t for t in out["train"] if t["utrip_id"] == "B_1")
    assert b["cities"][0] != b["cities"][1], "consecutive dup not collapsed"
    assert len(b["cities"]) == 4, "B_1 should be 4 after collapse"
    t = out["test"][0]
    print("test prefix:", t["cities"], "target:", t["target_city"], "query:", t["query"])
    assert t["target_city"] >= N_RESERVED, "target 40 was in train -> real id >= 2"
    assert min(min(x["cities"]) for x in out["train"]) >= N_RESERVED, \
        "real city ids must start at 2 (0=PAD,1=UNK)"
    # --- merge_train_val: leave-last-out trips come back as full trips -------
    full = next(t for t in out["train"] if t["utrip_id"] == "A_1")
    val = _to_leave_last_out(full)
    fake = {**out, "valid": [val], "meta": dict(out["meta"])}
    m = merge_train_val(fake)
    back = m["train"][-1]
    assert back["cities"] == full["cities"], "restored valid trip != original"
    assert back["month"] == full["month"] and back["los"] == full["los"]
    assert back["pos_index"] == full["pos_index"] and back["query"] is None
    assert "target_city" not in back and m["valid"] == []
    assert len(m["train"]) == len(out["train"]) + 1 and len(fake["train"]) == len(out["train"])
    assert m["test"] is fake["test"], "test must be untouched"
    print("merge_train_val OK")
    print("ALL SMOKE-TEST ASSERTIONS PASSED")
