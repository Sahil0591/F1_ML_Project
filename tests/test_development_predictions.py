"""Historical development exports are coherent, complete, and label free."""

import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.models.development import publish_development_fold


def test_development_export_removes_outcomes_and_checks_joint_probabilities(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    cutoff = datetime(2026, 9, 1, tzinfo=UTC)
    event = "season=2026/round=15"
    drivers = ["driver_a", "driver_b", "driver_c"]
    pq.write_table(
        pa.Table.from_pylist([{"event_id": event, "driver_id": name} for name in drivers]),
        benchmark / "gold.parquet",
    )
    manifest = {
        "datasets": {
            "Gold": {
                "path": "gold.parquet",
                "sha256": file_sha256(benchmark / "gold.parquet"),
            }
        }
    }
    (benchmark / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    predictions = tmp_path / "outer.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "event_id": event,
                    "driver_id": name,
                    "prediction_timestamp": cutoff,
                    "cutoff_kind": "post_qualifying",
                    "backend": "catboost",
                    "winner_probability": float(rank == 0),
                    "podium_probability": 1.0,
                    "dnf_probability": 0.1,
                    "finish_distribution": [float(index == rank) for index in range(3)],
                    "expected_position": float(rank + 1),
                    "label_winner": rank == 0,
                }
                for rank, name in enumerate(drivers)
            ]
        ),
        predictions,
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "benchmark_manifest_sha256": file_sha256(benchmark / "manifest.json"),
                "prediction_sha256": file_sha256(predictions),
                "tier": "Gold",
                "model_version": "joint-boosting-v2",
                "run_metadata": {"run_id": "test-run"},
                "folds": [
                    {
                        "backend": "catboost",
                        "event_id": event,
                        "status": "evaluated",
                        "fit_events": ["season=2026/round=01"],
                        "calibration_events": ["season=2026/round=02"],
                        "dnf_training_status": "fitted",
                        "calibration_label_availability_max": "2026-08-01T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "development.parquet"
    publication = publish_development_fold(
        report, predictions, benchmark, output, backend="catboost"
    )
    assert publication["status"] == "development_only"
    table = pq.read_table(output)
    assert not any(name.startswith("label_") for name in table.column_names)
    assert set(table["validation_status"].to_pylist()) == {"development_only"}
    assert sum(table["win_probability"].to_pylist()) == 1
    with pytest.raises(ValueError, match="already exists"):
        publish_development_fold(report, predictions, benchmark, output, backend="catboost")
