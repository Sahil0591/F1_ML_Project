"""Audit reliability and podium tail errors on paired outer Gold folds."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256


def _reliability(rows, probability, label):
    bins = []
    total = len(rows)
    ece = 0.0
    for index in range(10):
        selected = [
            row
            for row in rows
            if min(9, int(row[probability] * 10)) == index and row[label] is not None
        ]
        if not selected:
            continue
        mean_probability = sum(row[probability] for row in selected) / len(selected)
        observed = sum(row[label] for row in selected) / len(selected)
        bins.append(
            {
                "range": [index / 10, (index + 1) / 10],
                "count": len(selected),
                "predicted": mean_probability,
                "observed": observed,
            }
        )
        ece += len(selected) / total * abs(mean_probability - observed)
    return {"bins": bins, "expected_calibration_error": ece, "rows": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("baseline_predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if file_sha256(args.predictions) != report["prediction_sha256"]:
        raise ValueError("model prediction bytes changed")
    predictions = pq.read_table(args.predictions).to_pylist()
    paired = report["comparisons"]["post_qualifying"]
    events = set(next(iter(paired.values()))["paired_event_ids"])
    baseline = [
        row
        for row in pq.read_table(args.baseline_predictions).to_pylist()
        if row["event_id"] in events
    ]
    summary = {
        "model_report_sha256": file_sha256(args.report),
        "model_predictions_sha256": report["prediction_sha256"],
        "baseline_predictions_sha256": file_sha256(args.baseline_predictions),
        "paired_events": len(events),
        "models": {},
    }
    groups = {
        backend: [row for row in predictions if row["backend"] == backend] for backend in paired
    }
    groups["logistic"] = baseline
    for backend, rows in groups.items():
        if len({(row["event_id"], row["driver_id"]) for row in rows}) != len(rows):
            raise ValueError("calibration cohort has duplicate driver races")
        podium = "logistic_podium_probability" if backend == "logistic" else "podium_probability"
        winner = "logistic_winner_probability" if backend == "logistic" else "winner_probability"
        summary["models"][backend] = {
            "winner": _reliability(rows, winner, "label_winner"),
            "podium": _reliability(rows, podium, "label_podium"),
            "podium_high_confidence_predictions": sum(row[podium] >= 0.5 for row in rows),
            "podium_high_confidence_errors": [
                {"event_id": row["event_id"], "driver_id": row["driver_id"], "p": row[podium]}
                for row in rows
                if row[podium] >= 0.5 and not row["label_podium"]
            ],
            "podium_low_probability_hits": [
                {"event_id": row["event_id"], "driver_id": row["driver_id"], "p": row[podium]}
                for row in rows
                if row[podium] <= 0.05 and row["label_podium"]
            ],
            "podium_low_probability_predictions": sum(row[podium] <= 0.05 for row in rows),
        }
    if args.output.exists():
        raise ValueError("calibration audit already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    for backend, item in summary["models"].items():
        print(
            backend,
            "winner_ece",
            round(item["winner"]["expected_calibration_error"], 4),
            "podium_ece",
            round(item["podium"]["expected_calibration_error"], 4),
            "podium_high_errors",
            len(item["podium_high_confidence_errors"]),
            "/",
            item["podium_high_confidence_predictions"],
            "podium_low_hits",
            len(item["podium_low_probability_hits"]),
            "/",
            item["podium_low_probability_predictions"],
        )
    print(args.output)


if __name__ == "__main__":
    main()
