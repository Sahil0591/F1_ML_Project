"""Task-specific development race models sharing the existing joint race sampler.

The position strength comes from the existing joint boosting model. DNF comes
from a separate audited binary DNF model, so retirement risk is not a function
of pace. Both feed one Plackett-Luce/DNF sampler, which keeps winner, podium,
finish and DNF outputs coherent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from f1_ml_predictor.models.backtest import (
    RollingFold,
    _classifier,
    _cohort,
    _feature_columns,
    _heuristic_predictions,
    _matrix,
    _positive_probability,
    _probabilities_for_fold,
    _regressor,
    run_backtest,
)
from f1_ml_predictor.models.boosting import BACKENDS, estimator
from f1_ml_predictor.models.distributions import race_distribution, sample_race_orders
from f1_ml_predictor.models.probabilistic import (
    MODEL_VERSION,
    RaceModel,
    fit_race_model,
    run_probabilistic_backtest,
)
from f1_ml_predictor.models.protocol import PROTOCOL_SHA256
from f1_ml_predictor.prediction.history import GoldVersion
from f1_ml_predictor.prediction.live_features import apply_mask
from f1_ml_predictor.time import require_known_by
from f1_ml_predictor.trust.evidence import BenchmarkTier

COMPOSED_MODEL_VERSION = f"{MODEL_VERSION}+task-dnf-v1"
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _dnf_estimator(name: str, seed: int) -> Pipeline:
    if name == "logistic":
        return _classifier(seed)
    if name in BACKENDS:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", estimator(name, "dnf", seed, "cpu")),
            ]
        )
    raise ValueError(f"unsupported DNF model {name}")


def _complete_events(rows: list[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    """Keep earlier races only when every driver's label was known by the cutoff."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    return [
        row
        for group in groups.values()
        if all(
            item["prediction_timestamp"] < cutoff and item["label_available_at"] <= cutoff
            for item in group
        )
        for row in group
    ]


@dataclass
class DnfModel:
    name: str
    model: Pipeline | None
    prior: float
    columns: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
        if self.model is None:
            return np.full(len(rows), self.prior)
        projected = [{name: row.get(name) for name in self.columns} for row in rows]
        with threadpool_limits(limits=1):
            return np.asarray(
                _positive_probability(self.model, _matrix(projected, tuple(range(len(projected))))),
                dtype=float,
            )


def fit_dnf_model(rows: list[dict[str, Any]], name: str, seed: int, cutoff: datetime) -> DnfModel:
    """Fit only on complete earlier races with audited binary labels known at cutoff."""
    eligible = [row for row in _complete_events(rows, cutoff) if row["label_dnf"] is not None]
    columns = _feature_columns(rows)
    labels = [int(row["label_dnf"]) for row in eligible]
    prior = (1 + sum(labels)) / (2 + len(labels))
    metadata = {
        "model": name,
        "training_labels": len(labels),
        "training_retirements": sum(labels),
        "training_events": sorted({row["event_id"] for row in eligible}),
        "label_available_at_max": max(
            (row["label_available_at"] for row in eligible), default=None
        ),
        "feature_columns": list(columns),
    }
    if len(set(labels)) < 2:
        metadata["status"] = "smoothed_prior"
        return DnfModel(name, None, prior, columns, metadata)
    model = _dnf_estimator(name, seed)
    with threadpool_limits(limits=1):
        model.fit(_matrix(eligible, tuple(range(len(eligible)))), np.asarray(labels))
    metadata["status"] = "fitted"
    return DnfModel(name, model, prior, columns, metadata)


@dataclass
class ComposedRaceModel:
    position: RaceModel
    dnf: DnfModel
    temperature: float
    metadata: dict[str, Any]

    def outputs(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Any, Any]:
        ordered = sorted(rows, key=lambda row: row["driver_id"])
        position, _ = self.position.scores(ordered)
        return ordered, position * len(ordered), self.dnf.predict(ordered)

    def distribution(
        self, rows: list[dict[str, Any]], *, draws: int, seed: int
    ) -> list[dict[str, Any]]:
        ordered, position, dnf = self.outputs(rows)
        marginals = race_distribution(
            position.tolist(), dnf.tolist(), temperature=self.temperature, draws=draws, seed=seed
        )
        return [
            {
                "driver_id": row["driver_id"],
                "position_score": float(score),
                "dnf_model_probability": float(risk),
                **values,
            }
            for row, score, risk, values in zip(ordered, position, dnf, marginals, strict=True)
        ]

    def orders(
        self, rows: list[dict[str, Any]], *, draws: int, seed: int
    ) -> tuple[tuple[str, ...], ...]:
        ordered, position, dnf = self.outputs(rows)
        drivers = [row["driver_id"] for row in ordered]
        return tuple(
            tuple(drivers[index] for index in order)
            for order in sample_race_orders(
                position.tolist(),
                dnf.tolist(),
                temperature=self.temperature,
                draws=draws,
                seed=seed,
            )
        )


def _temperature(
    model: RaceModel, dnf: DnfModel, calibration: list[dict[str, Any]], seed: int
) -> tuple[float, list[tuple[float, float]]]:
    """Repeat the frozen temperature grid with the task-specific DNF composition."""
    groups: dict[tuple[str, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calibration:
        groups[_cohort(row)].append(row)
    losses = []
    for temperature in TEMPERATURES:
        event_losses = []
        for group in groups.values():
            ordered = sorted(group, key=lambda row: row["driver_id"])
            position, _ = model.scores(ordered)
            winner = next(index for index, row in enumerate(ordered) if row["label_winner"])
            probabilities = race_distribution(
                (position * len(ordered)).tolist(),
                dnf.predict(ordered).tolist(),
                temperature=temperature,
                draws=2048,
                seed=seed,
            )
            event_losses.append(-math.log(max(probabilities[winner]["winner_probability"], 1e-15)))
        losses.append((sum(event_losses) / len(event_losses), temperature))
    return min(losses)[1], losses


def fit_composed_model(
    gold_rows: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
    dnf_rows: list[dict[str, Any]],
    *,
    backend: str,
    dnf_name: str,
    cutoff: datetime,
    cutoff_kind: str,
    seed: int,
    device: str,
    hardware: dict[str, Any] | None,
) -> ComposedRaceModel:
    """Fit on every complete Gold race before the cutoff, then predict the live roster."""
    training = _complete_events(gold_rows, cutoff)
    for row in training:
        require_known_by(row["label_available_at"], cutoff)
    rows = [*training, *live_rows]
    fold = RollingFold(
        live_rows[0]["event_id"],
        cutoff_kind,
        cutoff,
        tuple(range(len(training))),
        tuple(range(len(training), len(rows))),
        tuple(sorted({row["event_id"] for row in training})),
    )
    position = fit_race_model(
        rows,
        fold,
        backend,
        min_fit_events=2,
        seed=seed,
        device=device,
        hardware=hardware,
        calibration_method="sigmoid",
        calibration_event_count=1,
    )
    if position is None:
        raise ValueError("insufficient Gold history to fit the position model")
    if datetime.fromisoformat(position.metadata["calibration_label_availability_max"]) > cutoff:
        raise ValueError("position model used labels published after the cutoff")
    calibration_cutoff = datetime.fromisoformat(
        position.metadata["calibration_prediction_timestamp"]
    )
    calibration_dnf = fit_dnf_model(dnf_rows, dnf_name, seed, calibration_cutoff)
    calibration_rows = [rows[index] for index in position.metadata["calibration_indices"]]
    temperature, losses = _temperature(position, calibration_dnf, calibration_rows, seed)
    dnf = fit_dnf_model(dnf_rows, dnf_name, seed, cutoff)
    metadata = {
        "model_version": COMPOSED_MODEL_VERSION,
        "position_backend": backend,
        "position_model": {
            key: value
            for key, value in position.metadata.items()
            if key not in {"fit_indices", "calibration_indices"}
        },
        "training_rows": len(training),
        "training_events": len(fold.train_events),
        "dnf_model": {
            **dnf.metadata,
            "label_available_at_max": str(dnf.metadata["label_available_at_max"]),
        },
        "calibration_dnf_model": {
            **calibration_dnf.metadata,
            "label_available_at_max": str(calibration_dnf.metadata["label_available_at_max"]),
        },
        "temperature": temperature,
        "temperature_validation_losses": losses,
        "temperature_policy": (
            "frozen grid on the latest calibration race with the task-specific DNF model "
            "fitted only on labels known at that race's cutoff"
        ),
    }
    return ComposedRaceModel(position, dnf, temperature, metadata)


def feature_importance(model: ComposedRaceModel, limit: int = 10) -> list[dict[str, Any]]:
    """Global split importance of the position model, where the backend exposes it."""
    estimator_values = getattr(model.position.position_model, "feature_importances_", None)
    if estimator_values is None:
        return []
    names = model.position.metadata["feature_columns"]
    values = np.asarray(estimator_values, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return []
    ranked = sorted(zip(names, values / total, strict=True), key=lambda item: -item[1])
    return [{"feature": name, "share": float(share)} for name, share in ranked[:limit] if share > 0]


def baseline_predictions(
    gold_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]], *, cutoff: datetime, seed: int
) -> dict[str, dict[str, Any]]:
    """Existing heuristic and logistic/Ridge baselines fitted on the same masked history."""
    training = _complete_events(gold_rows, cutoff)
    rows = [*training, *live_rows]
    fold = RollingFold(
        live_rows[0]["event_id"],
        live_rows[0]["cutoff_kind"],
        cutoff,
        tuple(range(len(training))),
        tuple(range(len(training), len(rows))),
        tuple(sorted({row["event_id"] for row in training})),
    )
    winner, winner_status = _probabilities_for_fold(rows, fold, "winner", seed)
    podium, podium_status = _probabilities_for_fold(rows, fold, "podium", seed)
    heuristic = _heuristic_predictions(rows, fold.test_indices)
    train = tuple(i for i in fold.train_indices if rows[i].get("label_position") is not None)
    regressor = _regressor()
    with threadpool_limits(limits=1):
        regressor.fit(_matrix(rows, train), np.asarray([rows[i]["label_position"] for i in train]))
        raw = regressor.predict(_matrix(rows, fold.test_indices))
    order = sorted(
        zip(fold.test_indices, raw, strict=True),
        key=lambda item: (item[1], rows[item[0]]["driver_id"]),
    )
    linear_rank = {index: rank for rank, (index, _) in enumerate(order, 1)}
    result: dict[str, dict[str, Any]] = {}
    for index in fold.test_indices:
        result[rows[index]["driver_id"]] = {
            "logistic_win_probability": winner.get(index),
            "logistic_podium_probability": podium.get(index),
            "linear_finish_rank": linear_rank[index],
            "heuristic_win_probability": heuristic[index]["winner"],
            "heuristic_finish_rank": heuristic[index]["position"],
        }
    result["_status"] = {"winner": winner_status, "podium": podium_status}
    return result


def evaluate_availability_variant(
    root: Path,
    version: GoldVersion,
    dnf_version: GoldVersion,
    masked: tuple[str, ...],
    *,
    backend: str,
    seed: int,
) -> dict[str, Any] | None:
    """Diagnostic frozen-protocol evaluation of the cutoff-matched feature set.

    It is cached by its exact inputs and is never selection evidence.
    """
    if not masked:
        return None
    dnf_masked = tuple(name for name in masked if name in dnf_version.feature_columns)
    key_payload = {
        "dataset_manifest_sha256": version.manifest_sha256,
        "dnf_dataset_manifest_sha256": dnf_version.manifest_sha256,
        "masked_features": sorted(masked),
        "backend": backend,
        "seed": seed,
        "draws": 4096,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    key = hashlib.sha256(_canonical(key_payload)).hexdigest()
    path = (
        root
        / "models/experiments/gold"
        / version.dataset_version
        / "availability_variants"
        / key
        / "evaluation.json"
    )
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("key") != key_payload:
            raise ValueError("cached availability evaluation does not match its inputs")
        return cached
    table = pa.Table.from_pylist(apply_mask(version.rows, masked), schema=version.schema)
    result = run_probabilistic_backtest(
        table,
        BenchmarkTier.GOLD,
        backends=(backend,),
        min_train_events=2,
        seed=seed,
        draws=4096,
        device="cpu",
        calibration_method="sigmoid",
        calibration_event_count=1,
    )
    dnf_table = pa.Table.from_pylist(
        apply_mask(dnf_version.rows, dnf_masked), schema=dnf_version.schema
    )
    dnf_result = run_backtest(dnf_table, BenchmarkTier.GOLD, min_train_events=3, seed=seed)
    evaluation = {
        "key": key_payload,
        "diagnostic_only": True,
        "selection_eligible": False,
        "purpose": "cutoff-matched feature availability check for development predictions",
        "position": {
            kind: {
                name: {
                    item: comparison[item]
                    for item in (
                        "metrics",
                        "baselines",
                        "paired_uncertainty",
                        "regressions",
                        "paired_event_ids",
                    )
                }
                for name, comparison in comparisons.items()
            }
            for kind, comparisons in result["comparisons"].items()
        },
        "dnf_baselines": {
            name: {
                task: {key: value for key, value in metrics.items() if key != "calibration"}
                for task, metrics in values.items()
                if task == "dnf"
            }
            for name, values in dnf_result.get("metrics", {}).items()
        },
        "dnf_evaluated_events": dnf_result.get("observation_counts", {}).get("evaluated_events"),
        "dnf_masked_features": list(dnf_masked),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json.dumps(evaluation, sort_keys=True, indent=2, allow_nan=False).encode())
    return evaluation
