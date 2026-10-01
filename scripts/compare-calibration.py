"""Compare two frozen calibration runs on identical outer race observations."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.models.uncertainty import paired_loss_intervals


def _reference(rows):
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
    parser.add_argument("sigmoid_report", type=Path)
    parser.add_argument("sigmoid_predictions", type=Path)
    parser.add_argument("isotonic_report", type=Path)
    parser.add_argument("isotonic_predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in (
            ("sigmoid", args.sigmoid_report),
            ("isotonic", args.isotonic_report),
        )
    }
    predictions = {}
    for name, path in (
        ("sigmoid", args.sigmoid_predictions),
        ("isotonic", args.isotonic_predictions),
    ):
        if file_sha256(path) != reports[name]["prediction_sha256"]:
            raise ValueError(f"{name} prediction bytes changed")
        predictions[name] = pq.read_table(path).to_pylist()
    if (
        reports["sigmoid"]["benchmark_manifest_sha256"]
        != reports["isotonic"]["benchmark_manifest_sha256"]
    ):
        raise ValueError("calibration runs use different benchmark versions")
    summary = {"benchmark_manifest_hash": reports["sigmoid"]["benchmark_manifest_sha256"]}
    for backend in ("hist", "catboost"):
        left = [row for row in predictions["sigmoid"] if row["backend"] == backend]
        right = [row for row in predictions["isotonic"] if row["backend"] == backend]

        def keys(rows):
            return {
                (row["event_id"], row["prediction_timestamp"], row["driver_id"]) for row in rows
            }

        if keys(left) != keys(right):
            raise ValueError("calibration runs use different outer race observations")
        summary[backend] = {
            "sigmoid_dnf": reports["sigmoid"]["comparisons"]["post_qualifying"][backend]["metrics"][
                "dnf"
            ],
            "isotonic_dnf": reports["isotonic"]["comparisons"]["post_qualifying"][backend][
                "metrics"
            ]["dnf"],
            "isotonic_minus_sigmoid": paired_loss_intervals(
                right, _reference(left), "logistic", seed=42
            ),
            "calibration_fit_counts": {
                name: {
                    status: sum(
                        fold.get("dnf_calibration", {}).get("status") == status
                        for fold in reports[name]["folds"]
                        if fold["backend"] == backend
                    )
                    for status in ("fitted", "skipped")
                }
                for name in ("sigmoid", "isotonic")
            },
        }
    if args.output.exists():
        raise ValueError("calibration comparison already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    for backend in ("hist", "catboost"):
        item = summary[backend]
        print(
            backend,
            "sigmoid",
            item["sigmoid_dnf"].get("brier_score"),
            "isotonic",
            item["isotonic_dnf"].get("brier_score"),
            "paired",
            item["isotonic_minus_sigmoid"]["dnf_brier"],
            "fits",
            item["calibration_fit_counts"],
        )
    print(args.output)


if __name__ == "__main__":
    main()
