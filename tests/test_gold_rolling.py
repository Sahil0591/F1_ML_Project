import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.benchmarks.rolling import ROLLING_FEATURE_COLUMNS, build_gold_rolling
from f1_ml_predictor.models.backtest import _feature_columns


def _fixture(root: Path, *, delayed: bool = False, gap: bool = False) -> None:
    benchmark = root / "data/benchmarks/gold_core"
    benchmark.mkdir(parents=True)
    registry = root / "data/benchmarks/gold_core_registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    races = []
    rows = []
    for round_number in (1, 2, 4) if gap else (1, 2, 3, 4):
        event = f"season=2025/round={round_number:02d}"
        cutoff = datetime(2025, 3, round_number * 3, 12, tzinfo=UTC)
        label_at = cutoff + timedelta(days=1)
        if delayed and round_number == 2:
            label_at = datetime(2025, 3, 15, tzinfo=UTC)
        rows.append(
            {
                "event_id": event,
                "driver_id": "driver_a",
                "prediction_timestamp": cutoff,
                "benchmark_tier": "Gold",
                "cutoff_kind": "post_qualifying",
                "label_final_audited": True,
                "label_audit_reference": f"fia:{round_number}",
                "label_available_at": label_at,
                "label_position": round_number,
                "label_dnf": round_number == 2,
            }
        )
        outcome = root / "data" / f"outcome-{round_number}.bin"
        outcome.write_bytes(f"audited-{round_number}".encode())
        races.append(
            {
                "event_id": event,
                "outcomes": {
                    "path": outcome.relative_to(root).as_posix(),
                    "sha256": file_sha256(outcome),
                },
            }
        )
    registry.write_text(json.dumps({"races": races}), encoding="utf-8")
    gold = benchmark / "gold.parquet"
    pq.write_table(pa.Table.from_pylist(rows), gold)
    datasets = {"Gold": {"path": "gold.parquet", "sha256": file_sha256(gold), "rows": len(rows)}}
    for tier in ("Silver", "Development"):
        path = benchmark / f"{tier.lower()}.parquet"
        pq.write_table(pa.Table.from_pylist([], schema=pq.read_schema(gold)), path)
        datasets[tier] = {"path": path.name, "sha256": file_sha256(path), "rows": 0}
    coverage = benchmark / "coverage.json"
    coverage.write_text("{}", encoding="utf-8")
    (benchmark / "manifest.json").write_text(
        json.dumps(
            {
                "catalog_sha256": file_sha256(registry),
                "coverage_sha256": file_sha256(coverage),
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )


def test_rolling_uses_only_contiguous_prior_audited_races(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = build_gold_rolling(tmp_path)
    rows = pq.read_table(tmp_path / report["feature_path"]).to_pylist()
    fourth = next(row for row in rows if row["event_id"].endswith("round=04"))
    assert fourth["recent_finish_mean_3"] == 2.0
    assert fourth["recent_dnf_rate_3"] == 1 / 3
    assert fourth["history_count_3"] == 3
    assert fourth["recent_finish_mean_5"] is None
    assert "round=04" not in fourth["provenance"]
    assert report["driver_race_observations"] == 4
    assert report["unique_drivers"] == 1
    enriched = pq.read_table(tmp_path / report["benchmark_dir"] / "gold.parquet").to_pylist()
    enriched_fourth = next(row for row in enriched if row["event_id"].endswith("round=04"))
    assert enriched_fourth["recent_finish_mean_3"] == 2.0
    assert enriched_fourth["recent_finish_mean_3_missing"] is False
    assert set(ROLLING_FEATURE_COLUMNS).issubset(_feature_columns(enriched))
    assert build_gold_rolling(tmp_path)["feature_sha256"] == report["feature_sha256"]


def test_rolling_rejects_missing_or_late_prior_race(tmp_path: Path) -> None:
    for scenario in ("gap", "delayed"):
        root = tmp_path / scenario
        _fixture(root, gap=scenario == "gap", delayed=scenario == "delayed")
        report = build_gold_rolling(root)
        rows = pq.read_table(root / report["feature_path"]).to_pylist()
        fourth = next(row for row in rows if row["event_id"].endswith("round=04"))
        assert fourth["recent_finish_mean_3"] is None
        assert fourth["recent_dnf_rate_3"] is None
        assert fourth["history_count_3"] == 0


def test_rolling_rejects_changed_gold_bytes(tmp_path: Path) -> None:
    _fixture(tmp_path)
    gold = tmp_path / "data/benchmarks/gold_core/gold.parquet"
    gold.write_bytes(gold.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="benchmark bytes"):
        build_gold_rolling(tmp_path)
