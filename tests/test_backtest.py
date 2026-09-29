from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS
from f1_ml_predictor.features.snapshot import FEATURE_SCHEMA, NUMERIC_FEATURES
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.models.backtest import rolling_folds, run_backtest, run_backtest_files
from f1_ml_predictor.trust.evidence import BenchmarkTier, EvidenceClass
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION


def benchmark_table(event_count: int = 6) -> pa.Table:
    feature_rows = []
    label_rows = []
    for round_number in range(1, event_count + 1):
        event = EventId(2025, round_number)
        cutoff = datetime(2025, 3, 1, 10, tzinfo=UTC) + timedelta(days=7 * round_number)
        evidence = {
            "class": EvidenceClass.CAPTURED_LIVE.value,
            "reference": f"capture-{round_number}",
            "available_at": (cutoff - timedelta(minutes=5)).isoformat(),
            "captured_at": (cutoff - timedelta(minutes=5)).isoformat(),
            "artifact_sha256": hashlib.sha256(f"table-{round_number}".encode()).hexdigest(),
            "source_published_at": None,
            "archive_version": None,
            "reconstruction_method": None,
            "audited": False,
            "tier": BenchmarkTier.GOLD.value,
        }
        for driver_number, driver_id in enumerate(
            ("driver_a", "driver_b", "driver_c", "driver_d"), start=1
        ):
            feature = {field.name: None for field in FEATURE_SCHEMA}
            for field in FEATURE_SCHEMA:
                if field.name.endswith("_missing"):
                    feature[field.name] = True
            feature.update(
                {
                    "event_id": event.partition(),
                    "driver_id": driver_id,
                    "constructor_id": "team_x" if driver_number < 3 else "team_y",
                    "circuit_id": "silverstone",
                    "prediction_timestamp": cutoff,
                    "feature_timestamp": cutoff - timedelta(minutes=5),
                    "feature_version": "2",
                    "history_count": 0,
                    "form_window": 5,
                    "provenance": "{}",
                    "qualifying_position": float(driver_number),
                    "qualifying_position_missing": False,
                    "grid_position": float(driver_number),
                    "grid_position_missing": False,
                    "recent_dnf_rate": 0.1 * driver_number,
                    "recent_dnf_rate_missing": False,
                    "benchmark_tier": BenchmarkTier.GOLD.value,
                    "cutoff_kind": "post_qualifying",
                    "qualifying_status": "completed",
                    "start_type": "grid",
                    "pit_lane_start": False,
                    "grid_status": "provisional",
                }
            )
            feature["feature_evidence"] = json.dumps(
                {
                    name: {
                        "missing": name
                        not in {"qualifying_position", "grid_position", "recent_dnf_rate"},
                        "tier": BenchmarkTier.GOLD.value
                        if name in {"qualifying_position", "grid_position", "recent_dnf_rate"}
                        else None,
                        "inputs": [{"reference": evidence["reference"], "evidence": evidence}]
                        if name in {"qualifying_position", "grid_position", "recent_dnf_rate"}
                        else [],
                    }
                    for name in NUMERIC_FEATURES
                }
            )
            feature_rows.append(feature)
            winner = driver_number == 1
            classified = driver_number != 4
            position = driver_number if classified else None
            label_rows.append(
                {
                    "event_id": event.partition(),
                    "driver_id": driver_id,
                    "constructor_id": feature["constructor_id"],
                    "circuit_id": "silverstone",
                    "prediction_timestamp": cutoff,
                    "feature_timestamp": feature["feature_timestamp"],
                    "cutoff_kind": "post_qualifying",
                    "benchmark_tier": BenchmarkTier.GOLD.value,
                    **{name: feature.get(name) for name in BENCHMARK_FEATURE_COLUMNS},
                    "label_season": event.season,
                    "label_round": event.round,
                    "label_driver_id": driver_id,
                    "label_position": position,
                    "label_classified": classified,
                    "label_winner": winner,
                    "label_podium": classified and position is not None and position <= 3,
                    "label_dnf": driver_number == 4,
                    "label_dnf_category": "retired_mechanical"
                    if driver_number == 4
                    else "finished",
                    "label_raw_status": "Retired" if driver_number == 4 else "Finished",
                    "label_taxonomy_version": DNF_TAXONOMY_VERSION,
                    "label_final_audited": True,
                    "label_available_at": cutoff + timedelta(days=2),
                    "label_audit_reference": f"fia-final-{round_number}",
                }
            )
    return pa.Table.from_pylist(label_rows)


def test_empty_gold_returns_insufficient_data_without_accuracy_metrics() -> None:
    table = pa.table(
        {
            "benchmark_tier": pa.array([], type=pa.string()),
        }
    )
    result = run_backtest(table, BenchmarkTier.GOLD)
    assert result["status"] == "insufficient_data"
    assert result["metrics"] == {}
    assert result["predictions"] == []


def test_empty_local_gold_backtest_writes_honest_report_and_empty_predictions(tmp_path) -> None:
    from f1_ml_predictor.benchmarks.builder import build_benchmarks

    benchmark_dir = tmp_path / "data" / "benchmarks"
    build_benchmarks(tmp_path, benchmark_dir)
    report_path = tmp_path / "models" / "backtests" / "gold.json"
    predictions_path = tmp_path / "data" / "predictions" / "backtests" / "gold.parquet"
    result = run_backtest_files(
        benchmark_dir / "gold.parquet",
        BenchmarkTier.GOLD,
        report_path,
        predictions_path,
    )
    assert result["status"] == "insufficient_data"
    assert result["prediction_rows"] == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["metrics"] == {}
    assert pq.read_table(predictions_path).num_rows == 0


def test_rolling_backtest_is_deterministic_and_probabilities_are_coherent() -> None:
    table = benchmark_table()
    first = run_backtest(table, BenchmarkTier.GOLD, min_train_events=2, seed=17)
    second = run_backtest(table, BenchmarkTier.GOLD, min_train_events=2, seed=17)
    assert first["status"] == "evaluated"
    assert first["primary_accuracy_claim_allowed"] is False
    assert len(first["folds"]) == 4
    assert first["predictions"] == second["predictions"]
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["logistic"]["winner"]["status"] == "evaluated"
    assert first["metrics"]["logistic"]["podium"]["status"] == "evaluated"
    assert first["metrics"]["logistic"]["dnf"]["status"] == "evaluated"
    grouped = {}
    for row in first["predictions"]:
        key = (row["event_id"], row["prediction_timestamp"])
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        assert sum(row["logistic_winner_probability"] for row in rows) == pytest.approx(1.0)
        assert sum(row["logistic_podium_probability"] for row in rows) == pytest.approx(3.0)
        assert all(0.0 <= row["logistic_podium_probability"] <= 1.0 for row in rows)


def test_fold_training_respects_label_availability_and_never_splits_race(tmp_path) -> None:
    rows = benchmark_table(6).to_pylist()
    delayed_event = EventId(2025, 3).partition()
    for row in rows:
        if row["event_id"] == delayed_event:
            row["label_available_at"] = datetime(2025, 5, 15, tzinfo=UTC)
    folds, skipped = rolling_folds(rows, min_train_events=3)
    event_four = EventId(2025, 4).partition()
    assert event_four not in {fold.event_id for fold in folds}
    assert any(item["event_id"] == event_four for item in skipped)
    for fold in folds:
        train_events = {rows[index]["event_id"] for index in fold.train_indices}
        test_events = {rows[index]["event_id"] for index in fold.test_indices}
        assert len(test_events) == 1
        assert fold.event_id not in train_events
        assert all(
            rows[index]["label_available_at"] <= fold.prediction_timestamp
            for index in fold.train_indices
        )


def test_development_evaluation_cannot_make_primary_claim() -> None:
    rows = benchmark_table(5).to_pylist()
    for row in rows:
        row["benchmark_tier"] = BenchmarkTier.DEVELOPMENT.value
    table = pa.Table.from_pylist(rows)
    result = run_backtest(table, BenchmarkTier.DEVELOPMENT)
    assert result["status"] == "evaluated"
    assert result["primary_accuracy_claim_allowed"] is False
