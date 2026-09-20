"""
download_data.py -- ONE-TIME data fetch (development convenience only).

This script is NOT part of the model pipeline. The graded code (src/ + the
notebooks) only ever reads the CSVs from the local `data/` folder; it never
references a URL. Run this once to populate data/, or copy the CSVs in by hand.

Usage:
    python scripts/download_data.py            # -> ./data/
    python scripts/download_data.py --out data
"""

import argparse
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main"
FILES = ["train_set.csv", "test_set.csv", "ground_truth.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in FILES:
        dst = os.path.join(args.out, name)
        if os.path.exists(dst):
            print(f"[skip] {dst} already exists")
            continue
        url = f"{BASE}/{name}"
        print(f"[get ] {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)
    print("Done. Files in", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
