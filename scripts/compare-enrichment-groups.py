"""Paired race intervals for single-group masks against one masked baseline."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.models.uncertainty import paired_loss_intervals


def _reference(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "logistic_winner_probability": row["winner_probability"],
            "logistic_podium_probability": row["podium_probability"],
            "logistic_dnf_probability": row["dnf_probability"],
            "linear_finish_position": row["expected_position"],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    baseline = pq.read_table(root / summary["stages"]["rolling_baseline"]["prediction"]).to_pylist()
    comparisons = {}
    for stage in ("constructor_only", "practice_only", "grid_only"):
        rows = pq.read_table(root / summary["stages"][stage]["prediction"]).to_pylist()
        comparisons[stage] = {
            backend: paired_loss_intervals(
                [row for row in rows if row["backend"] == backend],
                _reference([row for row in baseline if row["backend"] == backend]),
                "logistic",
                seed=42,
            )
            for backend in summary["backends"]
        }
    output = args.summary.parent / "paired_single_groups.json"
    content = json.dumps(
        {"version": 1, "diagnostic_only": True, "group_minus_masked_baseline": comparisons},
        sort_keys=True,
        indent=2,
    ).encode()
    if output.exists() and output.read_bytes() != content:
        raise ValueError("immutable paired group comparison collision")
    if not output.exists():
        output.write_bytes(content)
    print(output.resolve().relative_to(root).as_posix())


if __name__ == "__main__":
    main()
