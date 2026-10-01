"""Report nonmissing predictor counts for a versioned benchmark."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.benchmark / "manifest.json").read_text(encoding="utf-8"))
    rows = pq.read_table(args.benchmark / "gold.parquet").to_pylist()
    print("races", len(manifest["datasets"]["Gold"]["events"]), "rows", len(rows))
    for feature in manifest["feature_columns"]:
        if feature.endswith("_missing"):
            continue
        count = sum(row.get(feature) is not None for row in rows)
        print(feature, count, "/", len(rows))
    print("known_binary_dnf", sum(row["label_dnf"] is not None for row in rows))


if __name__ == "__main__":
    main()
