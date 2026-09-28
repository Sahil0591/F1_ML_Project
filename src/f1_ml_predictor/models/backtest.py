"""Deterministic heuristics and logistic baselines with race-grouped rolling origins."""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS, file_sha256
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.evidence import BenchmarkTier

_CLASSIFICATION_TASKS = {
    "winner": "label_winner",
    "podium": "label_podium",
    "dnf": "label_dnf",
}
_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("prediction_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("feature_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("cutoff_kind", pa.string(), nullable=False),
        pa.field("benchmark_tier", pa.string(), nullable=False),
        pa.field("heuristic_winner_probability", pa.float64()),
        pa.field("logistic_winner_probability", pa.float64()),
        pa.field("heuristic_podium_probability", pa.float64()),
        pa.field("logistic_podium_probability", pa.float64()),
        pa.field("heuristic_dnf_probability", pa.float64()),
        pa.field("logistic_dnf_probability", pa.float64()),
        pa.field("grid_finish_position", pa.float64()),
        pa.field("linear_finish_position", pa.float64()),
        pa.field("label_winner", pa.bool_(), nullable=False),
        pa.field("label_podium", pa.bool_(), nullable=False),
        pa.field("label_dnf", pa.bool_()),
        pa.field("label_position", pa.int64()),
        pa.field("label_available_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class RollingFold:
    event_id: str
    cutoff_kind: str
    prediction_timestamp: datetime
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_events: tuple[str, ...]


def _cohort(row: dict[str, Any]) -> tuple[str, datetime, str]:
    return row["event_id"], row["prediction_timestamp"], row["cutoff_kind"]


def rolling_folds(
    rows: list[dict[str, Any]], *, min_train_events: int = 2
) -> tuple[list[RollingFold], list[dict[str, Any]]]:
    """Create chronological test cohorts with complete event fields and available labels."""
    if (
        isinstance(min_train_events, bool)
        or not isinstance(min_train_events, int)
        or min_train_events < 1
    ):
        raise ValueError("min_train_events must be positive")
    cohorts: dict[tuple[str, datetime, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        require_utc(row["prediction_timestamp"], "prediction_timestamp")
        require_utc(row["label_available_at"], "label_available_at")
        require_known_by(row["feature_timestamp"], row["prediction_timestamp"])
        require_known_by(row["prediction_timestamp"], row["label_available_at"])
        cohorts[_cohort(row)].append(index)

    ordered = sorted(cohorts, key=lambda cohort: (cohort[1], cohort[0], cohort[2]))
    folds, skipped = [], []
    for event_id, cutoff, cutoff_kind in ordered:
        test_indices = tuple(cohorts[(event_id, cutoff, cutoff_kind)])
        eligible: dict[str, tuple[datetime, list[int]]] = {}
        for candidate, indices in cohorts.items():
            prior_event, prior_cutoff, prior_kind = candidate
            if prior_event == event_id or prior_kind != cutoff_kind or prior_cutoff >= cutoff:
                continue
            known = [index for index in indices if rows[index]["label_available_at"] <= cutoff]
            if known:
                current = eligible.get(prior_event)
                if current is None or prior_cutoff > current[0]:
                    eligible[prior_event] = (prior_cutoff, known)
        train_events = tuple(sorted(eligible, key=lambda event: eligible[event][0]))
        train_indices = tuple(index for event in train_events for index in eligible[event][1])
        if len(train_events) < min_train_events:
            skipped.append(
                {
                    "event_id": event_id,
                    "cutoff_kind": cutoff_kind,
                    "prediction_timestamp": cutoff.isoformat(),
                    "reason": "insufficient_prior_labelled_events",
                    "available_train_events": len(train_events),
                }
            )
            continue
        if set(train_events) & {event_id}:
            raise ValueError("rolling fold leaked the test event into training")
        folds.append(
            RollingFold(event_id, cutoff_kind, cutoff, train_indices, test_indices, train_events)
        )
    return folds, skipped


def _matrix(rows: list[dict[str, Any]], indices: tuple[int, ...]) -> np.ndarray[Any, Any]:
    return np.asarray(
        [
            [
                float(rows[index][name]) if rows[index].get(name) is not None else np.nan
                for name in BENCHMARK_FEATURE_COLUMNS
            ]
            for index in indices
        ],
        dtype=np.float64,
    )


def _classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(l1_ratio=0.0, max_iter=1000, random_state=seed),
            ),
        ]
    )


def _regressor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _positive_probability(model: Pipeline, matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    estimator = model.named_steps["model"]
    probabilities = cast(np.ndarray[Any, Any], model.predict_proba(matrix))
    positive_index = list(estimator.classes_).index(True)
    return probabilities[:, positive_index]


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def _positive_value(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or value <= 0:
        return None
    return float(value)


def _at_most_k(values: list[float], places: int) -> list[float]:
    count = len(values)
    target = min(places, count)
    if count == 0:
        return []
    if all(value <= 0 for value in values):
        return [target / count] * count
    low, high = 0.0, max(1.0, target / min(value for value in values if value > 0))
    for _ in range(80):
        scale = (low + high) / 2.0
        if sum(min(1.0, value * scale) for value in values) < target:
            low = scale
        else:
            high = scale
    return [min(1.0, value * high) for value in values]


def _group_indices(rows: list[dict[str, Any]], indices: tuple[int, ...]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[rows[index]["event_id"]].append(index)
    return groups


def _heuristic_predictions(
    rows: list[dict[str, Any]], indices: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    predictions: dict[int, dict[str, float]] = {}
    for group in _group_indices(rows, indices).values():
        ordered = sorted(group, key=lambda index: rows[index]["driver_id"])
        scores = []
        positions = []
        for index in ordered:
            row = rows[index]
            position = next(
                (
                    value
                    for name in ("grid_position", "qualifying_position", "recent_finish_mean")
                    if (value := _positive_value(row, name)) is not None
                ),
                None,
            )
            positions.append(float(position) if position is not None else float(len(group) + 1))
            scores.append(1.0 / positions[-1])
        winner = _normalize(scores)
        podium_order = sorted(
            range(len(ordered)), key=lambda i: (positions[i], rows[ordered[i]]["driver_id"])
        )
        podium = [1.0 if rank in podium_order[:3] else 0.0 for rank in range(len(ordered))]
        dnf = [
            min(1.0, max(0.0, float(rows[index]["recent_dnf_rate"])))
            if rows[index].get("recent_dnf_rate") is not None
            else 0.0
            for index in ordered
        ]
        for rank, index in enumerate(ordered):
            predictions[index] = {
                "winner": winner[rank],
                "podium": podium[rank],
                "dnf": dnf[rank],
                "position": float(podium_order.index(rank) + 1),
            }
    return predictions


def _calibration(
    y_true: list[bool], probability: list[float], bins: int = 10
) -> list[dict[str, Any]]:
    bucket: dict[int, list[tuple[bool, float]]] = defaultdict(list)
    for actual, predicted in zip(y_true, probability, strict=True):
        index = min(int(predicted * bins), bins - 1)
        bucket[index].append((actual, predicted))
    return [
        {
            "count": len(values),
            "mean_predicted": sum(value for _, value in values) / len(values),
            "observed_rate": sum(actual for actual, _ in values) / len(values),
        }
        for _, values in sorted(bucket.items())
    ]


def _binary_metrics(
    rows: list[dict[str, Any]], probability_field: str, label_field: str
) -> dict[str, Any]:
    known = [
        row
        for row in rows
        if row.get(label_field) is not None and row.get(probability_field) is not None
    ]
    if not known or len({row[label_field] for row in known}) < 2:
        return {"status": "insufficient_data", "n": len(known)}
    labels = [bool(row[label_field]) for row in known]
    probabilities = [float(row[probability_field]) for row in known]
    return {
        "status": "evaluated",
        "n": len(known),
        "log_loss": float(log_loss(labels, probabilities, labels=[False, True])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "calibration": _calibration(labels, probabilities),
    }


def _winner_metrics(rows: list[dict[str, Any]], probability_field: str) -> dict[str, Any]:
    groups: dict[tuple[str, datetime, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(probability_field) is not None:
            groups[(row["event_id"], row["prediction_timestamp"], row["cutoff_kind"])].append(row)
    if not groups:
        return {"status": "insufficient_data", "n": 0}
    losses, briers, top_one, top_three = [], [], [], []
    for race in groups.values():
        if sum(row["label_winner"] for row in race) != 1:
            raise ValueError("winner evaluation race must have one winner")
        ordered = sorted(race, key=lambda row: (-row[probability_field], row["driver_id"]))
        actual = next(row for row in race if row["label_winner"])
        p = min(1.0, max(1e-15, float(actual[probability_field])))
        losses.append(-math.log(p))
        briers.append(
            sum((float(row[probability_field]) - float(row["label_winner"])) ** 2 for row in race)
        )
        top_one.append(ordered[0]["label_winner"])
        top_three.append(any(row["label_winner"] for row in ordered[:3]))
    return {
        "status": "evaluated",
        "n": len(groups),
        "log_loss": sum(losses) / len(losses),
        "brier_score": sum(briers) / len(briers),
        "top_1_accuracy": sum(top_one) / len(top_one),
        "top_3_accuracy": sum(top_three) / len(top_three),
    }


def _position_metrics(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    known = [
        row
        for row in rows
        if row.get("label_position") is not None and row.get(prediction_field) is not None
    ]
    if not known:
        return {"status": "insufficient_data", "n": 0}
    return {
        "status": "evaluated",
        "n": len(known),
        "mean_absolute_error": float(
            mean_absolute_error(
                [row["label_position"] for row in known],
                [row[prediction_field] for row in known],
            )
        ),
    }


def _probabilities_for_fold(
    rows: list[dict[str, Any]], fold: RollingFold, task: str, seed: int
) -> tuple[dict[int, float], str]:
    target = _CLASSIFICATION_TASKS[task]
    train = tuple(index for index in fold.train_indices if rows[index].get(target) is not None)
    if len({rows[index][target] for index in train}) < 2:
        return {}, "insufficient_training_classes"
    model = _classifier(seed)
    with threadpool_limits(limits=1):
        model.fit(_matrix(rows, train), np.asarray([rows[index][target] for index in train]))
        raw = _positive_probability(model, _matrix(rows, fold.test_indices))
    by_index = {index: float(value) for index, value in zip(fold.test_indices, raw, strict=True)}
    if task == "winner":
        for group in _group_indices(rows, fold.test_indices).values():
            normalized = _normalize([by_index[index] for index in group])
            by_index.update({index: value for index, value in zip(group, normalized, strict=True)})
    elif task == "podium":
        for group in _group_indices(rows, fold.test_indices).values():
            projected = _at_most_k([by_index[index] for index in group], 3)
            by_index.update({index: value for index, value in zip(group, projected, strict=True)})
    return by_index, "evaluated"


def run_backtest(
    table: pa.Table,
    tier: BenchmarkTier,
    *,
    min_train_events: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run grouped rolling-origin heuristics and logistic/regression baselines."""
    if not isinstance(tier, BenchmarkTier):
        raise ValueError("tier must be explicit")
    rows = table.to_pylist()
    if not rows:
        return {
            "status": "insufficient_data",
            "reason": "benchmark dataset is empty",
            "tier": tier.value,
            "folds": [],
            "skipped_cohorts": [],
            "metrics": {},
            "predictions": [],
        }
    if not set(BENCHMARK_FEATURE_COLUMNS).issubset(table.column_names):
        raise ValueError("benchmark dataset is missing declared predictor columns")
    required_labels = {
        "label_winner",
        "label_podium",
        "label_dnf",
        "label_position",
        "label_available_at",
    }
    if not required_labels.issubset(table.column_names):
        raise ValueError("benchmark dataset is missing outcome labels")
    if set(table["benchmark_tier"].to_pylist()) != {tier.value}:
        raise ValueError("benchmark dataset tier does not match the requested tier")
    if any(name.startswith("label_") for name in BENCHMARK_FEATURE_COLUMNS):
        raise ValueError("target labels cannot be predictor columns")
    folds, skipped = rolling_folds(rows, min_train_events=min_train_events)
    prediction_rows: list[dict[str, Any]] = []
    fold_reports = []
    heuristic_metrics: list[dict[str, Any]] = []
    model_metrics: list[dict[str, Any]] = []
    for fold in folds:
        heuristic = _heuristic_predictions(rows, fold.test_indices)
        model_predictions: dict[int, dict[str, float | None]] = {
            index: {task: None for task in _CLASSIFICATION_TASKS} for index in fold.test_indices
        }
        task_status = {}
        for task in _CLASSIFICATION_TASKS:
            values, status = _probabilities_for_fold(rows, fold, task, seed)
            task_status[task] = status
            for index, value in values.items():
                model_predictions[index][task] = value
        regression_train = tuple(
            index for index in fold.train_indices if rows[index].get("label_position") is not None
        )
        position_status = "insufficient_training_labels"
        if len(regression_train) >= 2:
            regressor = _regressor()
            with threadpool_limits(limits=1):
                regressor.fit(
                    _matrix(rows, regression_train),
                    np.asarray([rows[index]["label_position"] for index in regression_train]),
                )
                raw_position = regressor.predict(_matrix(rows, fold.test_indices))
            for index, value in zip(fold.test_indices, raw_position, strict=True):
                model_predictions[index]["position"] = max(
                    1.0, min(float(len(fold.test_indices)), float(value))
                )
            position_status = "evaluated"
        for group in _group_indices(rows, fold.test_indices).values():
            order = sorted(
                group,
                key=lambda index: (
                    model_predictions[index].get("position")
                    if model_predictions[index].get("position") is not None
                    else heuristic[index]["position"],
                    rows[index]["driver_id"],
                ),
            )
            model_rank = {index: float(rank + 1) for rank, index in enumerate(order)}
            for index in group:
                feature = rows[index]
                prediction_rows.append(
                    {
                        "event_id": feature["event_id"],
                        "driver_id": feature["driver_id"],
                        "prediction_timestamp": feature["prediction_timestamp"],
                        "feature_timestamp": feature["feature_timestamp"],
                        "cutoff_kind": feature["cutoff_kind"],
                        "benchmark_tier": tier.value,
                        "heuristic_winner_probability": heuristic[index]["winner"],
                        "logistic_winner_probability": model_predictions[index]["winner"],
                        "heuristic_podium_probability": heuristic[index]["podium"],
                        "logistic_podium_probability": model_predictions[index]["podium"],
                        "heuristic_dnf_probability": heuristic[index]["dnf"],
                        "logistic_dnf_probability": model_predictions[index]["dnf"],
                        "grid_finish_position": heuristic[index]["position"],
                        "linear_finish_position": model_rank[index]
                        if position_status == "evaluated"
                        else None,
                        "label_winner": feature["label_winner"],
                        "label_podium": feature["label_podium"],
                        "label_dnf": feature["label_dnf"],
                        "label_position": feature["label_position"],
                        "label_available_at": feature["label_available_at"],
                    }
                )
        fold_reports.append(
            {
                "event_id": fold.event_id,
                "cutoff_kind": fold.cutoff_kind,
                "prediction_timestamp": fold.prediction_timestamp.isoformat(),
                "train_events": list(fold.train_events),
                "test_rows": len(fold.test_indices),
                "model_status": {**task_status, "position": position_status},
            }
        )
        heuristic_metrics.extend(
            row
            for row in prediction_rows
            if row["event_id"] == fold.event_id
            and row["prediction_timestamp"] == fold.prediction_timestamp
            and row["cutoff_kind"] == fold.cutoff_kind
        )
        model_metrics.extend(
            row
            for row in prediction_rows
            if row["event_id"] == fold.event_id
            and row["prediction_timestamp"] == fold.prediction_timestamp
            and row["cutoff_kind"] == fold.cutoff_kind
        )

    if not folds:
        return {
            "status": "insufficient_data",
            "reason": "no cohort has enough earlier labelled events for a rolling origin",
            "tier": tier.value,
            "folds": [],
            "skipped_cohorts": skipped,
            "metrics": {},
            "predictions": [],
        }
    metrics = {
        "heuristic": {
            "winner": _winner_metrics(heuristic_metrics, "heuristic_winner_probability"),
            "podium": _binary_metrics(
                heuristic_metrics, "heuristic_podium_probability", "label_podium"
            ),
            "dnf": _binary_metrics(heuristic_metrics, "heuristic_dnf_probability", "label_dnf"),
            "finishing_position": _position_metrics(heuristic_metrics, "grid_finish_position"),
        },
        "logistic": {
            "winner": _winner_metrics(model_metrics, "logistic_winner_probability"),
            "podium": _binary_metrics(model_metrics, "logistic_podium_probability", "label_podium"),
            "dnf": _binary_metrics(model_metrics, "logistic_dnf_probability", "label_dnf"),
            "finishing_position": _position_metrics(model_metrics, "linear_finish_position"),
        },
    }
    return {
        "status": "evaluated",
        "tier": tier.value,
        "primary_accuracy_claim_allowed": tier == BenchmarkTier.GOLD,
        "seed": seed,
        "scikit_learn_version": sklearn.__version__,
        "estimators": {
            "classification": "LogisticRegression(l1_ratio=0.0, max_iter=1000, solver=lbfgs)",
            "position": "Ridge(alpha=1.0)",
            "numeric_preprocessing": "fold-median imputation and standard scaling",
        },
        "feature_columns": list(BENCHMARK_FEATURE_COLUMNS),
        "folds": fold_reports,
        "skipped_cohorts": skipped,
        "metrics": metrics,
        "predictions": prediction_rows,
    }


def run_backtest_files(
    dataset_path: Path,
    tier: BenchmarkTier,
    report_path: Path,
    prediction_path: Path,
    *,
    min_train_events: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    table = pq.read_table(dataset_path)
    result = run_backtest(table, tier, min_train_events=min_train_events, seed=seed)
    manifest_path = dataset_path.parent / "manifest.json"
    result["benchmark_dataset_sha256"] = file_sha256(dataset_path)
    result["benchmark_manifest_sha256"] = file_sha256(manifest_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = result.pop("predictions")
    result["prediction_rows"] = len(predictions)
    report_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(predictions, schema=_OUTPUT_SCHEMA), prediction_path)
    return result


def prediction_content_hash(table: pa.Table) -> str:
    """Stable semantic digest used by deterministic smoke tests."""
    rows = sorted(
        json.dumps(row, sort_keys=True, default=lambda value: value.isoformat(), allow_nan=False)
        for row in table.to_pylist()
    )
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()
