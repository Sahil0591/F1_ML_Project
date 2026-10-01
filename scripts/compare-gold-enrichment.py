"""Pair a new Gold model run with the preserved 95-race source run."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.models.uncertainty import paired_loss_intervals

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for name in (
        "previous_report",
        "previous_predictions",
        "enriched_report",
        "enriched_predictions",
    ):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    previous = json.loads(args.previous_report.read_text(encoding="utf-8"))
    enriched = json.loads(args.enriched_report.read_text(encoding="utf-8"))
    if (
        sha(args.previous_predictions) != previous["prediction_sha256"]
        or sha(args.enriched_predictions) != enriched["prediction_sha256"]
    ):
        raise ValueError("prediction bytes differ from a frozen report")
    previous_rows = pq.read_table(args.previous_predictions).to_pylist()
    enriched_rows = pq.read_table(args.enriched_predictions).to_pylist()
    results = {}
    for backend in ("hist", "catboost"):
        old = previous["comparisons"]["post_qualifying"][backend]
        new = enriched["comparisons"]["post_qualifying"][backend]
        if old["paired_event_ids"] != new["paired_event_ids"]:
            raise ValueError("historical and enriched runs have different outer race folds")
        old_rows = [row for row in previous_rows if row["backend"] == backend]
        new_rows = [row for row in enriched_rows if row["backend"] == backend]
        results[backend] = {
            "paired_events": len(old["paired_event_ids"]),
            "previous_metrics": old["metrics"],
            "enriched_metrics": new["metrics"],
            "enriched_minus_previous_race_loss": paired_loss_intervals(
                new_rows, _reference(old_rows), "logistic", seed=42
            ),
        }
    result = {
        "version": 1,
        "diagnostic_only": True,
        "protocol": "gold-chronological-v2",
        "previous_manifest_sha256": previous["benchmark_manifest_sha256"],
        "enriched_manifest_sha256": enriched["benchmark_manifest_sha256"],
        "previous_report": args.previous_report.resolve().relative_to(ROOT).as_posix(),
        "enriched_report": args.enriched_report.resolve().relative_to(ROOT).as_posix(),
        "models": results,
    }
    output = (
        ROOT
        / "models/experiments/gold/enrichment_comparison"
        / enriched["benchmark_dataset_sha256"]
        / "comparison.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result, sort_keys=True, indent=2).encode()
    if output.exists() and output.read_bytes() != content:
        raise ValueError("immutable enrichment comparison collision")
    if not output.exists():
        output.write_bytes(content)
    print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
