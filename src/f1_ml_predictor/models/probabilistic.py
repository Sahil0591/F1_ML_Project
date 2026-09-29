"""Chronological calibration and controlled comparisons of joint race models."""

import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS, file_sha256
from f1_ml_predictor.models.backtest import (
    RollingFold,
    _binary_metrics,
    _cohort,
    _feature_columns,
    _matrix,
    _position_metrics,
    _winner_metrics,
    rolling_folds,
    run_backtest,
)
from f1_ml_predictor.models.boosting import BACKENDS, estimator
from f1_ml_predictor.models.calibration import calibrated_probabilities, fit_binary_calibrator
from f1_ml_predictor.models.distributions import race_distribution
from f1_ml_predictor.models.hardware import choose_device, library_versions
from f1_ml_predictor.models.protocol import (
    PRELIMINARY_PAIRED_EVENTS,
    PROTOCOL,
    PROTOCOL_SHA256,
    SELECTION_PAIRED_EVENTS,
)
from f1_ml_predictor.models.uncertainty import paired_loss_intervals
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.evidence import BenchmarkTier

MODEL_VERSION = "joint-boosting-v2"
_PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string()),
        pa.field("driver_id", pa.string()),
        pa.field("prediction_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("feature_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("cutoff_kind", pa.string()),
        pa.field("benchmark_tier", pa.string()),
        pa.field("backend", pa.string()),
        pa.field("model_version", pa.string()),
        pa.field("training_device", pa.string()),
        pa.field("winner_probability", pa.float64()),
        pa.field("podium_probability", pa.float64()),
        pa.field("dnf_probability", pa.float64()),
        pa.field("finish_distribution", pa.list_(pa.float64())),
        pa.field("expected_position", pa.float64()),
        pa.field("simulation_standard_error_max", pa.float64()),
        pa.field("label_winner", pa.bool_()),
        pa.field("label_podium", pa.bool_()),
        pa.field("label_dnf", pa.bool_()),
        pa.field("label_position", pa.int64()),
        pa.field("label_available_at", pa.timestamp("us", tz="UTC")),
    ]
)


def _estimator_configuration(model: Any) -> dict[str, Any]:
    """Preserve a backend's explicit NaN missing-value sentinel in strict JSON."""
    parameters = model.get_params()
    encoded = {
        key: {"type": "float", "value": "NaN"}
        if isinstance(value, float) and math.isnan(value)
        else value
        for key, value in parameters.items()
    }
    json.dumps(encoded, allow_nan=False)
    return encoded


@dataclass
class RaceModel:
    backend: str
    device: str
    seed: int
    imputer: Any
    position_model: Any
    dnf_model: Any
    dnf_prior: float
    calibrator: Any
    temperature: float
    metadata: dict[str, Any]

    def scores(self, rows: list[dict[str, Any]]) -> tuple[Any, Any]:
        matrix = self.imputer.transform(_matrix(rows, tuple(range(len(rows)))))
        with threadpool_limits(limits=1):
            position = np.asarray(self.position_model.predict(matrix), dtype=float)
            dnf = np.full(len(rows), self.dnf_prior)
            if self.dnf_model is not None:
                positive = list(self.dnf_model.classes_).index(1)
                dnf = self.dnf_model.predict_proba(matrix)[:, positive]
            if self.calibrator is not None:
                dnf = np.asarray(
                    calibrated_probabilities(
                        self.calibrator,
                        dnf.tolist(),
                        self.metadata.get("calibration_method", "sigmoid"),
                    )
                )
        return position, dnf

    def predict(self, rows: list[dict[str, Any]], *, draws: int = 4096) -> list[dict[str, Any]]:
        if not rows or len({_cohort(row) for row in rows}) != 1:
            raise ValueError("predict requires one complete race/cutoff roster")
        if len({row["driver_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate driver in prediction roster")
        for row in rows:
            require_utc(row["prediction_timestamp"], "prediction_timestamp")
            require_known_by(row["feature_timestamp"], row["prediction_timestamp"])
            if row["benchmark_tier"] != self.metadata["tier"]:
                raise ValueError("model and prediction evidence tiers differ")
            if row["cutoff_kind"] != self.metadata["cutoff_kind"]:
                raise ValueError("model and prediction cutoff cohorts differ")
            if row["event_id"] in (
                self.metadata["fit_events"] + self.metadata["calibration_events"]
            ):
                raise ValueError("prediction event was used to fit or calibrate the model")
            if row["prediction_timestamp"] < datetime.fromisoformat(
                self.metadata["calibration_label_availability_max"]
            ):
                raise ValueError("model contains labels unavailable at prediction cutoff")
        ordered = sorted(rows, key=lambda row: row["driver_id"])
        position, dnf = self.scores(ordered)
        probabilities = race_distribution(
            (position * len(rows)).tolist(),
            dnf.tolist(),
            temperature=self.temperature,
            draws=draws,
            seed=self.seed,
        )
        return [
            {
                **{
                    key: row[key]
                    for key in (
                        "event_id",
                        "driver_id",
                        "prediction_timestamp",
                        "feature_timestamp",
                        "cutoff_kind",
                        "benchmark_tier",
                    )
                },
                **probability,
                "backend": self.backend,
                "training_device": self.device,
                "model_version": MODEL_VERSION,
            }
            for row, probability in zip(ordered, probabilities, strict=True)
        ]


def calibration_split(
    rows: list[dict[str, Any]],
    fold: RollingFold,
    min_fit_events: int,
    calibration_event_count: int = 1,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(calibration_event_count) is not int or calibration_event_count < 1:
        raise ValueError("calibration event count must be a positive integer")
    groups: dict[str, list[int]] = defaultdict(list)
    for index in fold.train_indices:
        groups[rows[index]["event_id"]].append(index)
    if len(groups) < min_fit_events + 1:
        return (), ()
    ordered = sorted(groups, key=lambda event: rows[groups[event][0]]["prediction_timestamp"])
    count = min(calibration_event_count, len(ordered) - min_fit_events)
    calibration = tuple(index for event in ordered[-count:] for index in groups[event])
    calibration_cutoff = rows[calibration[0]]["prediction_timestamp"]
    eligible = [
        event
        for event in ordered[:-count]
        if all(
            rows[index]["label_available_at"] <= calibration_cutoff
            and rows[index]["prediction_timestamp"] < calibration_cutoff
            for index in groups[event]
        )
    ]
    if len(eligible) < min_fit_events:
        return (), ()
    return tuple(index for event in eligible for index in groups[event]), calibration


def fit_race_model(
    rows: list[dict[str, Any]],
    fold: RollingFold,
    backend: str,
    *,
    min_fit_events: int = 2,
    seed: int = 42,
    device: str = "cpu",
    hardware: dict[str, Any] | None = None,
    calibration_method: str = "sigmoid",
    calibration_event_count: int = 1,
) -> RaceModel | None:
    if calibration_method not in {"sigmoid", "isotonic", "identity"}:
        raise ValueError("unsupported calibration method")
    train, calibration = calibration_split(rows, fold, min_fit_events, calibration_event_count)
    if not train:
        return None
    position_train = tuple(i for i in train if rows[i].get("label_position") is not None)
    if len(position_train) < 2:
        return None
    chosen, reason = choose_device(backend, len(train), device, hardware)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    field_sizes: dict[tuple[str, Any, str], int] = defaultdict(int)
    for index in train:
        field_sizes[_cohort(rows[index])] += 1
    known_dnf = tuple(i for i in train if rows[i].get("label_dnf") is not None)
    prior = (1 + sum(bool(rows[i]["label_dnf"]) for i in known_dnf)) / (2 + len(known_dnf))

    def fit_models(training_device: str) -> tuple[Any, Any]:
        position_model = estimator(backend, "position", seed, training_device)
        position_model.fit(
            imputer.transform(_matrix(rows, position_train)),
            np.asarray(
                [rows[i]["label_position"] / field_sizes[_cohort(rows[i])] for i in position_train]
            ),
        )
        dnf_model = None
        if len({rows[i]["label_dnf"] for i in known_dnf}) == 2:
            dnf_model = estimator(backend, "dnf", seed, training_device)
            dnf_model.fit(
                imputer.transform(_matrix(rows, known_dnf)),
                np.asarray([int(rows[i]["label_dnf"]) for i in known_dnf]),
            )
        if backend == "xgboost" and training_device != "cpu":
            config = json.loads(position_model.get_booster().save_config())
            if not config["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("XGBoost silently fell back to CPU")
        return position_model, dnf_model

    start = time.perf_counter()
    with threadpool_limits(limits=1):
        imputer.fit(_matrix(rows, train))
        try:
            position_model, dnf_model = fit_models(chosen)
        except Exception as exc:
            if chosen == "cpu":
                raise
            reason = f"GPU training failed; CPU fallback: {type(exc).__name__}: {exc}"
            chosen = "cpu"
            position_model, dnf_model = fit_models(chosen)
    metadata = {
        "fit_events": sorted({rows[i]["event_id"] for i in train}),
        "calibration_events": sorted({rows[i]["event_id"] for i in calibration}),
        "fit_indices": list(train),
        "calibration_indices": list(calibration),
        "calibration_prediction_timestamp": rows[calibration[0]][
            "prediction_timestamp"
        ].isoformat(),
        "fit_label_availability_max": max(rows[i]["label_available_at"] for i in train).isoformat(),
        "calibration_label_availability_max": max(
            rows[i]["label_available_at"] for i in calibration
        ).isoformat(),
        "device": chosen,
        "device_reason": reason,
        "tier": rows[fold.test_indices[0]]["benchmark_tier"],
        "cutoff_kind": fold.cutoff_kind,
        "fit_seconds": time.perf_counter() - start,
        "dnf_training_labels": len(known_dnf),
        "dnf_training_status": "fitted"
        if dnf_model is not None
        else ("smoothed_prior" if known_dnf else "insufficient_data_prior"),
        "dnf_prior": prior,
        "reproducibility": "seeded CPU with one thread"
        if chosen == "cpu"
        else "seeded GPU; floating-point summation may vary",
        "configuration": _estimator_configuration(position_model),
        "feature_columns": list(_feature_columns(rows)),
        "feature_schema_version": "gold-rolling-v1"
        if len(_feature_columns(rows)) > len(BENCHMARK_FEATURE_COLUMNS)
        else "benchmark-feature-v2",
        "calibration_method": calibration_method,
        "calibration_event_count_requested": calibration_event_count,
    }
    model = RaceModel(
        backend, chosen, seed, imputer, position_model, dnf_model, prior, None, 1.0, metadata
    )
    calibration_rows = [rows[i] for i in calibration]
    _, raw_dnf = model.scores(calibration_rows)
    model.calibrator, calibration_metadata = fit_binary_calibrator(
        raw_dnf.tolist(),
        [row.get("label_dnf") for row in calibration_rows],
        method=calibration_method,
        seed=seed,
    )
    metadata["dnf_calibration"] = calibration_metadata
    metadata["dnf_calibration_status"] = (
        f"heldout_{calibration_method}"
        if model.calibrator is not None
        else (
            "identity_requested"
            if calibration_method == "identity"
            else "insufficient_samples_identity"
        )
    )
    losses = []
    groups: dict[tuple[str, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        groups[_cohort(row)].append(row)
    for temperature in (0.5, 1.0, 2.0, 4.0):
        event_losses = []
        for group in groups.values():
            ordered_calibration = sorted(group, key=lambda row: row["driver_id"])
            position, calibrated_dnf = model.scores(ordered_calibration)
            winner_index = next(
                i for i, row in enumerate(ordered_calibration) if row["label_winner"]
            )
            probabilities = race_distribution(
                (position * len(group)).tolist(),
                calibrated_dnf.tolist(),
                temperature=temperature,
                draws=2048,
                seed=seed,
            )
            probability = probabilities[winner_index]["winner_probability"]
            event_losses.append(float(-math.log(max(probability, 1e-15))))
        losses.append((sum(event_losses) / len(event_losses), temperature))
    _, model.temperature = min(losses)
    metadata["temperature"] = model.temperature
    metadata["temperature_validation_losses"] = losses
    metadata["calibration_policy"] = (
        "earlier complete events; fit labels known at earliest calibration cutoff"
    )
    return model


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "winner": _winner_metrics(rows, "winner_probability"),
        "podium": _binary_metrics(rows, "podium_probability", "label_podium"),
        "dnf": _binary_metrics(rows, "dnf_probability", "label_dnf"),
        "finishing_position": _position_metrics(rows, "expected_position"),
    }
    result["winner"]["calibration"] = _binary_metrics(
        rows, "winner_probability", "label_winner"
    ).get("calibration", [])
    known = [row for row in rows if row["label_position"] is not None]
    if known:
        result["finishing_position"]["ranked_probability_score"] = sum(
            sum(
                (sum(row["finish_distribution"][:k]) - float(row["label_position"] <= k)) ** 2
                for k in range(1, len(row["finish_distribution"]))
            )
            / max(1, len(row["finish_distribution"]) - 1)
            for row in known
        ) / len(known)
        grouped: dict[tuple[str, Any, str], list[dict[str, Any]]] = defaultdict(list)
        for row in known:
            grouped[_cohort(row)].append(row)
        correct, pairs = 0.0, 0
        for group in grouped.values():
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    difference = left["expected_position"] - right["expected_position"]
                    actual = left["label_position"] - right["label_position"]
                    correct += 0.5 if difference == 0 else float(difference * actual > 0)
                    pairs += 1
        result["finishing_position"]["pairwise_ranking_accuracy"] = (
            correct / pairs if pairs else None
        )
    return result


def _baseline_metrics(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return {
        "winner": _winner_metrics(rows, f"{model}_winner_probability"),
        "podium": _binary_metrics(rows, f"{model}_podium_probability", "label_podium"),
        "dnf": _binary_metrics(rows, f"{model}_dnf_probability", "label_dnf"),
        "finishing_position": _position_metrics(
            rows, "linear_finish_position" if model == "logistic" else "grid_finish_position"
        ),
    }


def select_candidate(
    comparisons: dict[str, Any],
    minimum_events: int = SELECTION_PAIRED_EVENTS,
    *,
    tier: BenchmarkTier = BenchmarkTier.DEVELOPMENT,
    eligible_events: int = 0,
) -> dict[str, Any]:
    """Select a provisional config only after paired coverage and regression checks."""
    if type(minimum_events) is not int or minimum_events < SELECTION_PAIRED_EVENTS:
        raise ValueError("model selection requires at least 25 paired Gold races")
    if (
        not isinstance(tier, BenchmarkTier)
        or type(eligible_events) is not int
        or eligible_events < 0
    ):
        raise ValueError("model selection needs an explicit tier and integer event count")
    if tier != BenchmarkTier.GOLD or eligible_events < minimum_events:
        return {
            "status": "deferred",
            "selected_backend": None,
            "reason": "requires at least 25 eligible Gold races for this cutoff",
            "minimum_gold_events": minimum_events,
            "eligible_gold_events": eligible_events if tier == BenchmarkTier.GOLD else 0,
        }
    eligible = []
    for backend, comparison in comparisons.items():
        metrics = comparison["metrics"]
        independent = len(set(comparison.get("paired_event_ids", [])))
        if independent >= minimum_events and not comparison["regressions"]:
            if all(task.get("status") == "evaluated" for task in metrics.values()):
                eligible.append((metrics["winner"]["log_loss"], backend))
    if not eligible:
        return {
            "status": "deferred",
            "selected_backend": None,
            "reason": "insufficient paired events or documented baseline regressions",
            "minimum_gold_events": minimum_events,
            "eligible_gold_events": eligible_events,
            "paired_events_by_backend": {
                backend: len(set(comparison.get("paired_event_ids", [])))
                for backend, comparison in comparisons.items()
            },
        }
    return {
        "status": "provisional",
        "selected_backend": min(eligible)[1],
        "reason": "lowest winner log loss among candidates passing paired baseline checks",
        "confirmation": (
            "requires future independent races; selection scores are not unbiased accuracy"
        ),
    }


def _validate_rows(table: pa.Table, tier: BenchmarkTier) -> list[dict[str, Any]]:
    if not isinstance(tier, BenchmarkTier):
        raise ValueError("tier must be explicit")
    required = set(BENCHMARK_FEATURE_COLUMNS) | {
        "event_id",
        "driver_id",
        "prediction_timestamp",
        "feature_timestamp",
        "cutoff_kind",
        "benchmark_tier",
        "label_winner",
        "label_podium",
        "label_dnf",
        "label_position",
        "label_available_at",
        "label_final_audited",
    }
    if not required.issubset(table.column_names):
        raise ValueError("benchmark is missing declared predictors, metadata, or audited labels")
    rows = sorted(table.to_pylist(), key=lambda row: (*_cohort(row), row["driver_id"]))
    groups: dict[tuple[str, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["benchmark_tier"] != tier.value or row["label_final_audited"] is not True:
            raise ValueError("benchmark tier mismatch or unaudited label")
        for key in ("label_winner", "label_podium"):
            if not isinstance(row[key], bool):
                raise ValueError("classification labels must be Boolean")
        if row["label_dnf"] is not None and not isinstance(row["label_dnf"], bool):
            raise ValueError("DNF labels must be Boolean or missing")
        groups[_cohort(row)].append(row)
    for group in groups.values():
        if len({row["driver_id"] for row in group}) != len(group):
            raise ValueError("duplicate driver in race/cutoff")
        if sum(row["label_winner"] for row in group) != 1:
            raise ValueError("race must have exactly one winner")
        positions = [row["label_position"] for row in group if row["label_position"] is not None]
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate observed finish position")
        for row in group:
            position = row["label_position"]
            if position is not None and (
                isinstance(position, bool)
                or not isinstance(position, int)
                or not 1 <= position <= len(group)
            ):
                raise ValueError("invalid observed finish position")
            if row["label_winner"] != (position == 1):
                raise ValueError("winner label and finish position disagree")
            if position is not None and row["label_podium"] != (position <= 3):
                raise ValueError("podium label and finish position disagree")
    if rows and np.isinf(_matrix(rows, tuple(range(len(rows))))).any():
        raise ValueError("infinite predictor values")
    return rows


def run_probabilistic_backtest(
    table: pa.Table,
    tier: BenchmarkTier,
    *,
    backends: tuple[str, ...] = BACKENDS,
    min_train_events: int = 2,
    seed: int = 42,
    draws: int = 4096,
    device: str = "cpu",
    hardware: dict[str, Any] | None = None,
    model_dir: Path | None = None,
    calibration_method: str = "sigmoid",
    calibration_event_count: int = 1,
) -> dict[str, Any]:
    if not backends or len(set(backends)) != len(backends) or set(backends) - set(BACKENDS):
        raise ValueError("choose distinct supported boosting backends")
    if device not in {"cpu", "auto", "cuda"}:
        raise ValueError("invalid device")
    if calibration_method not in {"sigmoid", "isotonic", "identity"}:
        raise ValueError("unsupported calibration method")
    if type(calibration_event_count) is not int or calibration_event_count < 1:
        raise ValueError("calibration event count must be a positive integer")
    if (
        isinstance(min_train_events, bool)
        or not isinstance(min_train_events, int)
        or min_train_events < 1
    ):
        raise ValueError("min_train_events must be a positive integer")
    race_distribution([1.0], [0.0], draws=draws, seed=seed)
    rows = _validate_rows(table, tier)
    folds, skipped = rolling_folds(rows, min_train_events=min_train_events + 1)
    ordered_table = pa.Table.from_pylist(rows, schema=table.schema)
    baseline = run_backtest(ordered_table, tier, min_train_events=min_train_events + 1, seed=seed)
    versions = library_versions()
    reports: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    availability: dict[str, Any] = {}
    for backend in backends:
        if backend != "hist" and versions[backend] is None:
            availability[backend] = {"status": "not_installed"}
            continue
        availability[backend] = {
            "status": "available",
            "version": versions.get("scikit-learn" if backend == "hist" else backend),
        }
        for fold in folds:
            model = fit_race_model(
                rows,
                fold,
                backend,
                min_fit_events=min_train_events,
                seed=seed,
                device=device,
                hardware=hardware,
                calibration_method=calibration_method,
                calibration_event_count=calibration_event_count,
            )
            if model is None:
                reports.append(
                    {
                        "backend": backend,
                        "event_id": fold.event_id,
                        "cutoff_kind": fold.cutoff_kind,
                        "prediction_timestamp": fold.prediction_timestamp.isoformat(),
                        "status": "insufficient_calibration_history",
                    }
                )
                continue
            prediction_rows = model.predict([rows[i] for i in fold.test_indices], draws=draws)
            labels = {rows[i]["driver_id"]: rows[i] for i in fold.test_indices}
            for prediction in prediction_rows:
                prediction.update(
                    {
                        key: labels[prediction["driver_id"]][key]
                        for key in (
                            "label_winner",
                            "label_podium",
                            "label_dnf",
                            "label_position",
                            "label_available_at",
                        )
                    }
                )
            predictions.extend(prediction_rows)
            report = {
                "backend": backend,
                "event_id": fold.event_id,
                "cutoff_kind": fold.cutoff_kind,
                "prediction_timestamp": fold.prediction_timestamp.isoformat(),
                "status": "evaluated",
                **model.metadata,
            }
            if model_dir is not None:
                key = hashlib.sha256(
                    json.dumps(
                        [
                            backend,
                            fold.event_id,
                            fold.cutoff_kind,
                            fold.prediction_timestamp.isoformat(),
                        ]
                    ).encode()
                ).hexdigest()[:20]
                path = model_dir / f"{key}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path)
                report["model_artifact"] = path.name
                report["model_sha256"] = file_sha256(path)
            reports.append(report)
    comparisons: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for kind in sorted({row["cutoff_kind"] for row in predictions}):
        by_backend = {
            backend: [
                row
                for row in predictions
                if row["backend"] == backend and row["cutoff_kind"] == kind
            ]
            for backend in backends
        }
        cohort_sets = [{_cohort(row) for row in group} for group in by_backend.values() if group]
        common = set.intersection(*cohort_sets) if cohort_sets else set()
        paired_baseline = [row for row in baseline["predictions"] if _cohort(row) in common]
        kind_comparisons = {}
        for backend, group in by_backend.items():
            if not group:
                continue
            metrics = _metrics([row for row in group if _cohort(row) in common])
            baselines = {
                name: _baseline_metrics(paired_baseline, name) for name in ("heuristic", "logistic")
            }
            regressions = []
            for name, baseline_metrics in baselines.items():
                for task, keys in {
                    "winner": ("log_loss", "brier_score", "top_1_accuracy", "top_3_accuracy"),
                    "podium": ("log_loss", "brier_score"),
                    "dnf": ("log_loss", "brier_score"),
                    "finishing_position": ("mean_absolute_error",),
                }.items():
                    for key in keys:
                        reference = baseline_metrics[task].get(key)
                        actual = metrics[task].get(key)
                        if reference is None or actual is None:
                            regressions.append(
                                {
                                    "baseline": name,
                                    "task": task,
                                    "metric": key,
                                    "reason": "metric unavailable",
                                }
                            )
                        elif (
                            actual < reference - 1e-12
                            if key.endswith("accuracy")
                            else actual > reference + 1e-12
                        ):
                            regressions.append(
                                {
                                    "baseline": name,
                                    "task": task,
                                    "metric": key,
                                    "delta": actual - reference,
                                }
                            )
            kind_comparisons[backend] = {
                "metrics": metrics,
                "baselines": baselines,
                "paired_uncertainty": {
                    name: paired_loss_intervals(
                        [row for row in group if _cohort(row) in common],
                        paired_baseline,
                        name,
                        seed=seed,
                    )
                    for name in ("heuristic", "logistic")
                },
                "regressions": regressions,
                "paired_cohorts": len(common),
                "paired_event_ids": sorted({cohort[0] for cohort in common}),
                "paired_driver_race_observations": len(
                    [row for row in group if _cohort(row) in common]
                ),
                "paired_unique_drivers": len(
                    {row["driver_id"] for row in group if _cohort(row) in common}
                ),
            }
        comparisons[kind] = kind_comparisons
        eligible_events = len({row["event_id"] for row in rows if row["cutoff_kind"] == kind})
        selections[kind] = select_candidate(
            kind_comparisons, tier=tier, eligible_events=eligible_events
        )
    return {
        "status": "evaluated" if predictions else "insufficient_data",
        "tier": tier.value,
        "evaluation_protocol": PROTOCOL,
        "evaluation_protocol_sha256": PROTOCOL_SHA256,
        "preliminary_comparison_allowed": tier == BenchmarkTier.GOLD
        and any(
            all(
                len(set(item["paired_event_ids"])) >= PRELIMINARY_PAIRED_EVENTS
                for item in comparison.values()
            )
            for comparison in comparisons.values()
            if comparison
        ),
        "primary_accuracy_claim_allowed": bool(predictions)
        and tier == BenchmarkTier.GOLD
        and any(
            all(
                len(set(item["paired_event_ids"])) >= PRELIMINARY_PAIRED_EVENTS
                and all(task.get("status") == "evaluated" for task in item["metrics"].values())
                for item in comparison.values()
            )
            for comparison in comparisons.values()
            if comparison
        ),
        "model_version": MODEL_VERSION,
        "seed": seed,
        "draws": draws,
        "libraries": versions,
        "observation_counts": {
            "eligible_driver_rows": len(rows),
            "eligible_driver_race_observations": len(rows),
            "unique_drivers": len({row["driver_id"] for row in rows}),
            "eligible_events_by_cutoff": {
                kind: len({row["event_id"] for row in rows if row["cutoff_kind"] == kind})
                for kind in sorted({row["cutoff_kind"] for row in rows})
            },
            "known_dnf_labels": sum(row["label_dnf"] is not None for row in rows),
        },
        "hardware": hardware,
        "requested_device": device,
        "feature_columns": list(_feature_columns(rows)),
        "backends": availability,
        "folds": reports,
        "skipped_cohorts": skipped,
        "comparisons": comparisons,
        "selection": selections
        or {
            "status": "deferred",
            "selected_backend": None,
            "reason": "no evaluable chronological calibration/test cohorts",
        },
        "predictions": predictions,
        "distribution_policy": (
            "independent sampled DNF; PL ordering within finishers and retirees; "
            "retirees trail finishers; finish distribution is total modeled order, "
            "not FIA classification"
        ),
        "limitations": (
            "independent DNF omits shared incidents; composition and finite sampling can "
            "change marginal calibration; evaluate final marginals on outer folds; "
            "DNF uses an unvalidated Beta(1,1) prior when no audited labels exist"
        ),
    }


def run_probabilistic_files(
    dataset_path: Path,
    tier: BenchmarkTier,
    report_path: Path,
    prediction_path: Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest_path = dataset_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["datasets"][tier.value]
    if record["path"] != dataset_path.name or record["sha256"] != file_sha256(dataset_path):
        raise ValueError("benchmark dataset does not match its manifest")
    if manifest["coverage_sha256"] != file_sha256(dataset_path.parent / "coverage.json"):
        raise ValueError("benchmark coverage does not match its manifest")
    table = pq.read_table(dataset_path)
    detected_features = _feature_columns(table.to_pylist())
    if manifest["feature_columns"] != list(detected_features):
        raise ValueError("benchmark predictor manifest mismatch")
    if len(detected_features) > len(BENCHMARK_FEATURE_COLUMNS) and (
        manifest.get("version") != 2 or manifest.get("rolling_version") != "gold-rolling-v1"
    ):
        raise ValueError("rolling predictors require a versioned benchmark manifest")
    if record["rows"] != table.num_rows:
        raise ValueError("benchmark row count mismatch")
    result = run_probabilistic_backtest(
        table, tier, model_dir=report_path.parent / "fitted", **settings
    )
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["git_commit"] = None
    result["source_sha256"] = {
        path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    result["benchmark_dataset_sha256"] = record["sha256"]
    result["benchmark_manifest_sha256"] = file_sha256(manifest_path)
    predictions = result.pop("predictions")
    result["prediction_rows"] = len(predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(predictions, schema=_PREDICTION_SCHEMA), prediction_path)
    result["prediction_sha256"] = file_sha256(prediction_path)
    report_path.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result
