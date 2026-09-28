from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import build_benchmarks, discover_local_races, file_sha256
from f1_ml_predictor.features.snapshot import CONSERVATIVE_ALIASES, FEATURE_SCHEMA, NUMERIC_FEATURES
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.evidence import BenchmarkTier, EvidenceClass
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION, OUTCOME_SCHEMA

UTC_TIME = datetime(2025, 3, 15, 10, tzinfo=UTC)


def feature_table(event: EventId, tier: BenchmarkTier) -> pa.Table:
    row = {field.name: None for field in FEATURE_SCHEMA}
    row.update(
        {
            "event_id": event.partition(),
            "driver_id": "driver_a",
            "constructor_id": "team_x",
            "circuit_id": "silverstone",
            "prediction_timestamp": UTC_TIME,
            "feature_timestamp": UTC_TIME - timedelta(minutes=5),
            "feature_version": "2",
            "history_count": 0,
            "form_window": 5,
            "provenance": "{}",
            "qualifying_position": 1.0,
            "qualifying_position_missing": False,
            "benchmark_tier": tier.value,
            "cutoff_kind": "post_qualifying",
            "qualifying_status": "completed",
            "start_type": "grid",
            "grid_status": "provisional",
        }
    )
    for field in FEATURE_SCHEMA:
        if field.name.endswith("_missing"):
            row[field.name] = True
    row["qualifying_position_missing"] = False
    evidence_kind = {
        BenchmarkTier.GOLD: EvidenceClass.CAPTURED_LIVE.value,
        BenchmarkTier.SILVER: EvidenceClass.CONSERVATIVE_RECONSTRUCTION.value,
        BenchmarkTier.DEVELOPMENT: EvidenceClass.CURRENT_STATE_ONLY.value,
    }[tier]
    proof = {
        "class": evidence_kind,
        "reference": "feature-input-proof",
        "available_at": UTC_TIME.isoformat() if tier != BenchmarkTier.DEVELOPMENT else None,
        "captured_at": UTC_TIME.isoformat() if tier == BenchmarkTier.GOLD else None,
        "artifact_sha256": hashlib.sha256(b"feature-input").hexdigest()
        if tier != BenchmarkTier.DEVELOPMENT
        else None,
        "source_published_at": None,
        "archive_version": None,
        "reconstruction_method": "method-v1" if tier == BenchmarkTier.SILVER else None,
        "audited": tier == BenchmarkTier.SILVER,
        "tier": tier.value,
    }
    input_ref = {"reference": "feature-input-proof", "evidence": proof}
    row["feature_evidence"] = json.dumps(
        {
            name: {
                "missing": name != "qualifying_position",
                "tier": tier.value if name == "qualifying_position" else None,
                "inputs": [input_ref] if name == "qualifying_position" else [],
            }
            for name in NUMERIC_FEATURES
        }
    )
    return pa.Table.from_pylist([row], schema=FEATURE_SCHEMA)


def outcome_table(
    event: EventId, *, available: datetime = UTC_TIME + timedelta(days=1)
) -> pa.Table:
    row = {
        "season": event.season,
        "round": event.round,
        "driver_id": "driver_a",
        "position": 1,
        "classified": True,
        "winner": True,
        "podium": True,
        "dnf": False,
        "dnf_category": "finished",
        "raw_status": "Finished",
        "taxonomy_version": DNF_TAXONOMY_VERSION,
        "final_audited": True,
        "label_available_at": available,
        "audit_reference": "fia-final-document",
    }
    return pa.Table.from_pylist([row], schema=OUTCOME_SCHEMA)


def write_input(root: Path, relative: str, table: pa.Table) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return {"path": relative, "sha256": file_sha256(path)}


def test_builds_three_tiers_with_audited_labels_and_manifest_hashes(tmp_path: Path) -> None:
    races = []
    for round_number, tier in enumerate(BenchmarkTier, start=1):
        event = EventId(2025, round_number)
        races.append(
            {
                "event_id": event.partition(),
                "prediction_timestamp": UTC_TIME.isoformat(),
                "cutoff_kind": "post_qualifying",
                "features": write_input(
                    tmp_path, f"inputs/features-{round_number}.parquet", feature_table(event, tier)
                ),
                "outcomes": write_input(
                    tmp_path, f"inputs/outcomes-{round_number}.parquet", outcome_table(event)
                ),
            }
        )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"version": 1, "races": races}), encoding="utf-8")

    first = build_benchmarks(tmp_path, tmp_path / "benchmarks", catalog)
    manifest = json.loads((tmp_path / "benchmarks" / "manifest.json").read_text())
    assert first["included_races"] == 3
    assert first["excluded_races"] == 0
    assert {tier: manifest["datasets"][tier]["rows"] for tier in manifest["datasets"]} == {
        "Gold": 1,
        "Silver": 1,
        "Development": 1,
    }
    assert "label_winner" in manifest["label_columns"]
    assert "label_available_at" in manifest["label_columns"]
    assert "label_label_available_at" not in manifest["label_columns"]
    assert "label_winner" not in manifest["feature_columns"]
    assert "practice_observed_best_lap_seconds" in manifest["feature_columns"]
    gold = pq.read_table(tmp_path / "benchmarks" / "gold.parquet")
    assert gold["practice_observed_best_lap_seconds"].type == pa.float64()
    assert gold["qualifying_position_missing"].type == pa.bool_()
    assert gold["pit_lane_start"].to_pylist() == [None]
    assert len(CONSERVATIVE_ALIASES) == 5
    assert manifest["catalog_sha256"] == file_sha256(catalog)
    coverage = json.loads((tmp_path / "benchmarks" / "coverage.json").read_text())
    assert (
        coverage["coverage"][0]["source_files"]["features"]["sha256"]
        == races[0]["features"]["sha256"]
    )
    assert pq.read_table(tmp_path / "benchmarks" / "gold.parquet")["label_winner"].to_pylist() == [
        True
    ]
    first_bytes = (tmp_path / "benchmarks" / "manifest.json").read_bytes()
    second = build_benchmarks(tmp_path, tmp_path / "benchmarks", catalog)
    assert second == first
    assert (tmp_path / "benchmarks" / "manifest.json").read_bytes() == first_bytes


def test_invalid_incomplete_field_and_early_labels_are_excluded(tmp_path: Path) -> None:
    event = EventId(2025, 1)
    feature_ref = write_input(
        tmp_path, "inputs/features.parquet", feature_table(event, BenchmarkTier.GOLD)
    )
    incomplete = pa.concat_tables([outcome_table(event), outcome_table(EventId(2025, 2))])
    outcome_ref = write_input(tmp_path, "inputs/outcomes.parquet", incomplete)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "version": 1,
                "races": [
                    {
                        "event_id": event.partition(),
                        "prediction_timestamp": UTC_TIME.isoformat(),
                        "cutoff_kind": "post_qualifying",
                        "features": feature_ref,
                        "outcomes": outcome_ref,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = build_benchmarks(tmp_path, tmp_path / "out", catalog)
    assert report["included_races"] == 0
    assert "outcomes do not match" in report["coverage"][0]["reasons"][0]

    early_ref = write_input(
        tmp_path,
        "inputs/early-outcomes.parquet",
        outcome_table(event, available=UTC_TIME),
    )
    catalog_value = json.loads(catalog.read_text(encoding="utf-8"))
    catalog_value["races"][0]["outcomes"] = early_ref
    catalog.write_text(json.dumps(catalog_value), encoding="utf-8")
    report = build_benchmarks(tmp_path, tmp_path / "out", catalog)
    assert "published after the prediction cutoff" in report["coverage"][0]["reasons"][0]


def test_local_discovery_reports_candidates_without_promoting_retrospective_inputs(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "data" / "normalized" / "season=2025" / "round=01" / "entries.parquet"
    entry.parent.mkdir(parents=True)
    pq.write_table(pa.table({"driver_id": ["driver_a"]}), entry)
    candidates = discover_local_races(tmp_path)
    assert [row["event_id"] for row in candidates] == ["season=2025/round=01"]
    report = build_benchmarks(tmp_path, tmp_path / "data" / "benchmarks")
    assert report["excluded_races"] == 1
    assert set(report["coverage"][0]["reasons"]) == {
        "feature_snapshot_not_registered",
        "audited_outcomes_not_registered",
    }
    tier_tables = [
        pq.read_table(tmp_path / "data" / "benchmarks" / f"{tier}.parquet")
        for tier in ("gold", "silver", "development")
    ]
    assert all(table.num_rows == 0 for table in tier_tables)
    assert tier_tables[0].schema.equals(tier_tables[1].schema)
    assert tier_tables[1].schema.equals(tier_tables[2].schema)


def test_tampered_catalog_hash_is_reported_as_excluded_race(tmp_path: Path) -> None:
    event = EventId(2025, 1)
    features = write_input(
        tmp_path, "inputs/features.parquet", feature_table(event, BenchmarkTier.GOLD)
    )
    outcomes = write_input(tmp_path, "inputs/outcomes.parquet", outcome_table(event))
    features["sha256"] = "0" * 64
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "version": 1,
                "races": [
                    {
                        "event_id": event.partition(),
                        "prediction_timestamp": UTC_TIME.isoformat(),
                        "cutoff_kind": "post_qualifying",
                        "features": features,
                        "outcomes": outcomes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = build_benchmarks(tmp_path, tmp_path / "out", catalog)
    assert "hash does not match" in report["coverage"][0]["reasons"][0]
