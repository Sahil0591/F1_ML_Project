"""Paired race-level diagnostics for the frozen Gold comparison and ablations."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "full": "4d99e8881f043ea9",
    "no_rolling": "a6948deab154bfca",
    "qualifying_only": "3faa167074c0567f",
}
REPORTS = ROOT / "models/experiments/gold"
PREDICTIONS = ROOT / "data/predictions/probabilistic"
BASELINES = ROOT / "data/predictions/backtests/dataset-4d99e8881f043ea9/gold.parquet"
BENCHMARK = (
    ROOT
    / "data/benchmarks/gold_core_rolling_v1"
    / "10568218180a38d886a06a701e35ae1c8260dc71376b4caa5b0ba030a30ff402"
    / "gold.parquet"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss(rows: list[dict], task: str, probability: str | None) -> dict[str, float | int]:
    known = [row for row in rows if row[f"label_{task}"] is not None]
    if not known:
        return {"n": 0}
    if task == "winner":
        winner = next(row for row in known if row["label_winner"])
        p = max(1e-12, min(1 - 1e-12, winner[probability]))
        return {
            "n": 1,
            "log_loss": -math.log(p),
            "brier_score": sum(
                (row[probability] - float(row["label_winner"])) ** 2 for row in known
            ),
        }
    if task == "position":
        return {
            "n": len(known),
            "mae": sum(abs(row[probability] - row["label_position"]) for row in known) / len(known),
        }
    probabilities = [max(1e-12, min(1 - 1e-12, row[probability])) for row in known]
    targets = [int(row[f"label_{task}"]) for row in known]
    return {
        "n": len(known),
        "log_loss": sum(
            -y * math.log(p) - (1 - y) * math.log(1 - p)
            for p, y in zip(probabilities, targets, strict=True)
        )
        / len(known),
        "brier_score": sum((p - y) ** 2 for p, y in zip(probabilities, targets, strict=True))
        / len(known),
    }


def bootstrap(deltas: list[float]) -> list[float]:
    if len(deltas) < 2:
        return [float("nan"), float("nan")]
    random = np.random.default_rng(42)
    array = np.asarray(deltas)
    draws = array[random.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return [round(float(value), 5) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    reports = {
        name: json.loads((REPORTS / f"dataset-{key}" / "comparison.json").read_text())
        for name, key in COHORTS.items()
    }
    predictions = {}
    for name, key in COHORTS.items():
        path = PREDICTIONS / f"dataset-{key}" / "gold.parquet"
        if digest(path) != reports[name]["prediction_sha256"]:
            raise ValueError(f"prediction bytes changed for {name}")
        predictions[name] = pq.read_table(path).to_pylist()
    baseline = pq.read_table(BASELINES).to_pylist()
    events = reports["full"]["comparisons"]["post_qualifying"]["hist"]["paired_event_ids"]
    if any(
        reports[name]["comparisons"]["post_qualifying"][backend]["paired_event_ids"] != events
        for name in COHORTS
        for backend in reports[name]["backends"]
    ):
        raise ValueError("ablation folds or backend cohorts differ")
    baseline_by_key = {(row["event_id"], row["driver_id"]): row for row in baseline}
    summary = {
        "version": 1,
        "cohorts": COHORTS,
        "paired_races": len(events),
        "events": events,
        "backends": {},
        "feature_missingness": {},
        "training_rows": {},
    }
    benchmark = pq.read_table(BENCHMARK).to_pylist()
    for feature in reports["full"]["feature_columns"]:
        summary["feature_missingness"][feature] = sum(row[feature] is None for row in benchmark)
    for backend in reports["full"]["backends"]:
        model_rows = {
            name: {
                (row["event_id"], row["driver_id"]): row
                for row in predictions[name]
                if row["backend"] == backend
            }
            for name in COHORTS
        }
        keys = sorted(key for key in model_rows["full"] if key[0] in events)
        if not all(
            set(keys) == {key for key in rows if key[0] in events} for rows in model_rows.values()
        ):
            raise ValueError("ablation driver-race keys differ")
        if not all(key in baseline_by_key for key in keys):
            raise ValueError("paired baseline key missing")
        by_event: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key in keys:
            by_event[key[0]].append(key)
        backend_report = {"aggregate": {}, "worst_races": {}, "fold_deltas": {}}
        for task, model_column, base_columns in (
            (
                "winner",
                "winner_probability",
                ("heuristic_winner_probability", "logistic_winner_probability"),
            ),
            (
                "podium",
                "podium_probability",
                ("heuristic_podium_probability", "logistic_podium_probability"),
            ),
            ("dnf", "dnf_probability", ("heuristic_dnf_probability", "logistic_dnf_probability")),
            ("position", "expected_position", ("grid_finish_position", "linear_finish_position")),
        ):
            metric = "mae" if task == "position" else "log_loss"
            per_event = []
            for event in events:
                event_keys = by_event[event]
                record = {"event_id": event}
                for name in COHORTS:
                    record[name] = loss(
                        [model_rows[name][key] for key in event_keys], task, model_column
                    )
                for label, column in zip(("heuristic", "logistic"), base_columns, strict=True):
                    record[label] = loss([baseline_by_key[key] for key in event_keys], task, column)
                per_event.append(record)
            for comparator in ("heuristic", "logistic", "no_rolling", "qualifying_only"):
                paired = [
                    row for row in per_event if metric in row["full"] and metric in row[comparator]
                ]
                deltas = [row["full"][metric] - row[comparator][metric] for row in paired]
                backend_report["aggregate"][f"{task}_{metric}_minus_{comparator}"] = {
                    "n_races": len(paired),
                    "mean_delta": round(float(np.mean(deltas)), 5) if deltas else None,
                    "bootstrap_95_percent": bootstrap(deltas),
                }
                if task != "position":
                    brier = [
                        row["full"]["brier_score"] - row[comparator]["brier_score"]
                        for row in paired
                    ]
                    backend_report["aggregate"][f"{task}_brier_score_minus_{comparator}"] = {
                        "n_races": len(paired),
                        "mean_delta": round(float(np.mean(brier)), 5) if brier else None,
                        "bootstrap_95_percent": bootstrap(brier),
                    }
            backend_report["worst_races"][task] = sorted(
                (
                    {
                        "event_id": row["event_id"],
                        "model": round(row["full"][metric], 5),
                        "logistic": round(row["logistic"][metric], 5),
                        "delta": round(row["full"][metric] - row["logistic"][metric], 5),
                    }
                    for row in per_event
                    if metric in row["full"] and metric in row["logistic"]
                ),
                key=lambda row: -row["delta"],
            )[:5]
            backend_report["fold_deltas"][task] = [
                {
                    "event_id": row["event_id"],
                    "delta_logistic": round(row["full"][metric] - row["logistic"][metric], 5),
                }
                for row in per_event
                if metric in row["full"] and metric in row["logistic"]
            ]
        folds = [
            fold
            for fold in reports["full"]["folds"]
            if fold.get("backend") == backend and fold.get("status") == "evaluated"
        ]
        summary["training_rows"][backend] = {
            "minimum": min(len(fold["fit_indices"]) for fold in folds),
            "maximum": max(len(fold["fit_indices"]) for fold in folds),
            "dnf_minimum": min(fold["dnf_training_labels"] for fold in folds),
            "dnf_maximum": max(fold["dnf_training_labels"] for fold in folds),
        }
        calibration = {}
        for task, column in (("podium", "podium_probability"), ("dnf", "dnf_probability")):
            bins = []
            for lower, upper in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.000001)):
                selected = [
                    row
                    for key, row in model_rows["full"].items()
                    if key[0] in events
                    and row[f"label_{task}"] is not None
                    and lower <= row[column] < upper
                ]
                if selected:
                    bins.append(
                        {
                            "range": [lower, upper],
                            "n": len(selected),
                            "mean_predicted": round(
                                float(np.mean([row[column] for row in selected])), 4
                            ),
                            "observed": round(
                                float(np.mean([row[f"label_{task}"] for row in selected])), 4
                            ),
                        }
                    )
            calibration[task] = bins
        top_choices = [
            max(
                (model_rows["full"][key] for key in by_event[event]),
                key=lambda row: row["winner_probability"],
            )
            for event in events
        ]
        calibration["winner_top_choice"] = {
            "mean_confidence": round(
                float(np.mean([row["winner_probability"] for row in top_choices])), 4
            ),
            "accuracy": round(float(np.mean([row["label_winner"] for row in top_choices])), 4),
        }
        backend_report["calibration"] = calibration
        backend_report["podium_failure_rows"] = {}
        for event in ("season=2025/round=12", "season=2025/round=21"):
            backend_report["podium_failure_rows"][event] = sorted(
                (
                    {
                        "driver_id": key[1],
                        "label": model_rows["full"][key]["label_podium"],
                        "model_p": round(model_rows["full"][key]["podium_probability"], 4),
                        "logistic_p": round(baseline_by_key[key]["logistic_podium_probability"], 4),
                    }
                    for key in by_event[event]
                ),
                key=lambda row: -row["model_p"],
            )
        summary["backends"][backend] = backend_report
    destination = REPORTS / "dataset-4d99e8881f043ea9" / "regression_audit.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {destination.relative_to(ROOT)}")
    print(f"paired races: {len(events)}")
    for backend, result in summary["backends"].items():
        print(
            backend,
            {
                key: value["mean_delta"]
                for key, value in result["aggregate"].items()
                if key.endswith("minus_logistic")
            },
        )
        print(
            "  intervals",
            {
                key: value["bootstrap_95_percent"]
                for key, value in result["aggregate"].items()
                if key.endswith("minus_logistic")
            },
        )
        print(
            "  ablations",
            {
                key: value["mean_delta"]
                for key, value in result["aggregate"].items()
                if key.endswith(("minus_no_rolling", "minus_qualifying_only"))
            },
        )
        print("  worst podium", result["worst_races"]["podium"][:3])
        print("  calibration", result["calibration"])
        if backend == "hist":
            for event, rows in result["podium_failure_rows"].items():
                print(
                    "  podium case",
                    event,
                    [row for row in rows if row["label"] or row["model_p"] > 0.5],
                )
    print("training", summary["training_rows"])
    print(
        "missingness", sorted(summary["feature_missingness"].items(), key=lambda row: -row[1])[:18]
    )


if __name__ == "__main__":
    main()
