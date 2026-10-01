"""Publish label-free historical development predictions from an outer fold."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256


def _check_race(rows: list[dict[str, Any]]) -> None:
    count = len(rows)
    if count < 3 or len({row["driver_id"] for row in rows}) != count:
        raise ValueError("development predictions need one complete race roster")
    if abs(sum(row["winner_probability"] for row in rows) - 1) > 1e-9:
        raise ValueError("winner probabilities do not sum to one")
    if abs(sum(row["podium_probability"] for row in rows) - 3) > 1e-9:
        raise ValueError("podium probabilities do not sum to three")
    for row in rows:
        distribution = row["finish_distribution"]
        if len(distribution) != count or any(not 0 <= value <= 1 for value in distribution):
            raise ValueError("invalid finishing position distribution")
        if abs(sum(distribution) - 1) > 1e-9:
            raise ValueError("finishing position distribution does not sum to one")
        if abs(distribution[0] - row["winner_probability"]) > 1e-9:
            raise ValueError("winner and finishing distribution disagree")
        if abs(sum(distribution[:3]) - row["podium_probability"]) > 1e-9:
            raise ValueError("podium and finishing distribution disagree")
        expected = sum(rank * value for rank, value in enumerate(distribution, 1))
        if not math.isclose(expected, row["expected_position"], abs_tol=1e-9):
            raise ValueError("expected finish and distribution disagree")
        if not 0 <= row["dnf_probability"] <= 1:
            raise ValueError("invalid DNF probability")
    for rank in range(count):
        if abs(sum(row["finish_distribution"][rank] for row in rows) - 1) > 1e-9:
            raise ValueError("finishing position is assigned to more than one driver")


def publish_development_fold(
    report_path: Path,
    prediction_path: Path,
    benchmark_dir: Path,
    output_path: Path,
    *,
    backend: str,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Export one held-out race without outcome labels or validation claims."""
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise ValueError("development prediction artifact already exists")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    benchmark_manifest_path = benchmark_dir / "manifest.json"
    if file_sha256(benchmark_manifest_path) != report["benchmark_manifest_sha256"]:
        raise ValueError("development benchmark manifest does not match the model run")
    manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    gold = manifest["datasets"]["Gold"]
    if file_sha256(benchmark_dir / gold["path"]) != gold["sha256"]:
        raise ValueError("development benchmark Gold bytes changed")
    if file_sha256(prediction_path) != report["prediction_sha256"]:
        raise ValueError("outer-fold predictions changed")
    if report["tier"] != "Gold" or report.get("diagnostic_only"):
        raise ValueError("development publisher requires a non-ablation Gold run")
    predictions = [
        row for row in pq.read_table(prediction_path).to_pylist() if row["backend"] == backend
    ]
    if not predictions:
        raise ValueError("requested backend has no outer-fold predictions")
    if event_id is None:
        event_id = max(
            {row["event_id"] for row in predictions},
            key=lambda event: max(
                row["prediction_timestamp"] for row in predictions if row["event_id"] == event
            ),
        )
    rows = [row for row in predictions if row["event_id"] == event_id]
    if len({(row["prediction_timestamp"], row["cutoff_kind"]) for row in rows}) != 1:
        raise ValueError("development export needs one named cutoff cohort")
    fold = next(
        (
            fold
            for fold in report["folds"]
            if fold["backend"] == backend
            and fold["event_id"] == event_id
            and fold["status"] == "evaluated"
        ),
        None,
    )
    if fold is None or event_id in fold["fit_events"] + fold["calibration_events"]:
        raise ValueError("development event was fitted or calibrated")
    if fold["dnf_training_status"] != "fitted":
        raise ValueError("development DNF output requires audited fitted labels")
    benchmark_rows = [
        row
        for row in pq.read_table(benchmark_dir / gold["path"]).to_pylist()
        if row["event_id"] == event_id
    ]
    if {row["driver_id"] for row in rows} != {row["driver_id"] for row in benchmark_rows}:
        raise ValueError("development output does not contain the complete Gold race roster")
    _check_race(rows)
    created_at = datetime.now(UTC)
    provenance = json.dumps(
        {
            "report_sha256": file_sha256(report_path),
            "outer_prediction_sha256": report["prediction_sha256"],
            "benchmark_manifest_sha256": report["benchmark_manifest_sha256"],
            "fitted_model_sha256": fold.get("model_sha256"),
            "fit_events": fold["fit_events"],
            "calibration_events": fold["calibration_events"],
            "calibration_label_availability_max": fold["calibration_label_availability_max"],
            "historical_heldout_event": True,
        },
        sort_keys=True,
    )
    published = [
        {
            "event_id": row["event_id"],
            "driver_id": row["driver_id"],
            "prediction_timestamp": row["prediction_timestamp"],
            "generated_at": created_at,
            "validation_status": "development_only",
            "win_probability": row["winner_probability"],
            "podium_probability": row["podium_probability"],
            "dnf_probability": row["dnf_probability"],
            "expected_finish": row["expected_position"],
            "finishing_position_distribution": row["finish_distribution"],
            "model_version": report["model_version"],
            "model_run_id": report["run_metadata"]["run_id"],
            "model_backend": backend,
            "dataset_manifest_hash": report["benchmark_manifest_sha256"],
            "provenance": provenance,
        }
        for row in sorted(rows, key=lambda item: item["driver_id"])
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(published), output_path)
    publication = {
        "status": "development_only",
        "historical_heldout_event": event_id,
        "model_run_id": report["run_metadata"]["run_id"],
        "backend": backend,
        "dataset_manifest_hash": report["benchmark_manifest_sha256"],
        "prediction_rows": len(published),
        "prediction_sha256": file_sha256(output_path),
        "created_at": created_at.isoformat(),
        "validated_forecast": False,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(publication, sort_keys=True, indent=2), encoding="utf-8"
    )
    return publication
