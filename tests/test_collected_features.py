import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.features.manifest import load_feature_request
from f1_ml_predictor.features.snapshot import build_snapshot
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.collected_features import attach_audited_outcomes, certify_capture
from f1_ml_predictor.trust.evidence import table_hash
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION, OUTCOME_SCHEMA
from f1_ml_predictor.trust.prospective import freeze_bundle, load_bundle, verify_bundle

CAPTURED = datetime(2026, 10, 3, 9, tzinfo=UTC)
EVENT = EventId(2026, 16)


def raw_bundle(root: Path, *, captured: datetime = CAPTURED, change: str | None = None) -> Path:
    race = {
        "season": "2026",
        "round": "16",
        "raceName": "Bahrain Grand Prix",
        "Circuit": {"circuitId": "sepang"},
        "date": "2026-10-04",
        "time": "09:00:00Z",
    }
    results = [
        {
            "position": str(position),
            "Driver": {"driverId": driver},
            "Constructor": {"constructorId": "team_a"},
            "Q1": "1:32.200",
            "Q2": "1:31.100",
        }
        for position, driver in enumerate(("driver_a", "driver_b"), start=1)
    ]
    if change == "constructor_missing":
        del results[0]["Constructor"]
    if change == "wrong_event":
        qualifying_race = {**race, "round": "17", "QualifyingResults": results}
    else:
        qualifying_race = {**race, "QualifyingResults": results}
    payloads = {
        "schedule": {
            "MRData": {"limit": "100", "offset": "0", "total": "1", "RaceTable": {"Races": [race]}}
        },
        "qualifying": {
            "MRData": {
                "limit": "100",
                "offset": "0",
                "total": "2",
                "RaceTable": {"Races": [qualifying_race]},
            }
        },
        "forecast": {"hourly": {"time": ["2026-10-04T09:00"], "temperature_2m": [30]}},
    }
    requests = {
        "schedule": {
            "role": "event",
            "url": "https://api.jolpi.ca/ergast/f1/2026/",
            "params": {"limit": 100, "offset": 0},
            "response_captured_at": captured.isoformat(),
        },
        "qualifying": {
            "role": "qualifying",
            "url": "https://api.jolpi.ca/ergast/f1/2026/16/qualifying/",
            "params": {"limit": 100, "offset": 0},
            "response_captured_at": captured.isoformat(),
        },
        "forecast": {
            "role": "forecast",
            "url": "https://api.open-meteo.com/v1/forecast",
            "params": {"latitude": 2.8, "longitude": 101.7, "hourly": "temperature_2m"},
            "response_captured_at": captured.isoformat(),
        },
    }
    if change == "missing_schedule_role":
        requests["schedule"]["role"] = "circuit"
    if change == "duplicate_qualifying_role":
        requests["forecast"]["role"] = "qualifying"
    if change == "wrong_url":
        requests["qualifying"]["url"] = "https://api.jolpi.ca/ergast/f1/2026/17/qualifying/"
    metadata = {
        "source_requests": requests,
        "window": {
            "race_start": "2026-10-04T09:00:00+00:00",
            "qualifying_decision_at": captured.isoformat(),
            "pre_race_minutes": 60,
        },
    }
    if change == "race_window_mismatch":
        metadata["window"]["race_start"] = "2026-10-04T10:00:00+00:00"
    tables = {
        name: pa.Table.from_pylist([{"payload_json": json.dumps(payload, sort_keys=True)}])
        for name, payload in payloads.items()
    }
    return freeze_bundle(
        root / "data/raw/prospective",
        EVENT,
        captured,
        "post_qualifying",
        captured,
        tables,
        metadata,
        now=captured,
    )


def registry(root: Path) -> dict[str, Any]:
    return json.loads((root / "data/benchmarks/prospective_registry.json").read_text())


def test_certifies_exact_retained_payloads_and_replays_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = raw_bundle(tmp_path)
    members = {path.name: path.read_bytes() for path in bundle.iterdir()}

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("certification must not request later data")

    monkeypatch.setattr(httpx.Client, "get", unexpected)
    result = certify_capture(tmp_path, bundle)
    assert result["gold_snapshot"] is True
    assert result["normalization_required"] is False
    assert result["evaluation_eligible"] is False
    features = pq.ParquetFile(tmp_path / result["features"]["path"]).read()
    rows = features.to_pylist()
    assert len(rows) == 2
    assert {row["benchmark_tier"] for row in rows} == {"Gold"}
    assert {row["prediction_timestamp"] for row in rows} == {CAPTURED}
    assert {row["feature_timestamp"] for row in rows} == {CAPTURED}
    assert {row["circuit_id"] for row in rows} == {"sepang"}
    assert [row["qualifying_position"] for row in rows] == [1.0, 2.0]
    assert [row["qualifying_last_session_seconds"] for row in rows] == [91.1, 91.1]
    assert [row["teammate_qualifying_position_delta"] for row in rows] == [-1.0, 1.0]
    for row in rows:
        assert row["grid_position"] is row["recent_finish_mean"] is None
        assert row["practice_best_seconds"] is row["forecast_temperature_2m"] is None
        assert row["grid_status"] == row["start_type"] == "unknown"
        assert row["history_count"] == 0
        assert row["forecast_temperature_2m_missing"] is True
    proof_path = tmp_path / result["feature_manifest"]["path"]
    proof = json.loads(proof_path.read_text())
    inputs, cutoff = load_feature_request(proof_path, tmp_path)
    replayed = build_snapshot(inputs, cutoff, certified_only=True)
    assert features.replace_schema_metadata(None).equals(replayed)
    assert inputs.rosters[0].evidence is not None
    assert inputs.rosters[0].evidence.artifact_sha256 == table_hash(inputs.rosters[0].table)
    raw_manifest, raw_tables = load_bundle(bundle)
    assert raw_manifest["manifest_sha256"] in inputs.rosters[0].evidence_reference
    raw_hash = hashlib.sha256(
        raw_tables["qualifying"]["payload_json"][0].as_py().encode()
    ).hexdigest()
    assert raw_hash in inputs.rosters[0].evidence_reference
    assert proof["unused_raw_inputs"] == ["forecast"]
    assert proof["raw_sources"]["forecast"]["params"]["hourly"] == "temperature_2m"
    assert members == {path.name: path.read_bytes() for path in bundle.iterdir()}
    verify_bundle(bundle)
    registered = registry(tmp_path)["races"][0]
    assert registered["features"] == result["features"]
    assert registered["outcomes"] is None
    assert registered["evaluation_eligible"] is False


def test_repeat_certification_is_idempotent_and_cannot_overwrite_features(tmp_path: Path) -> None:
    bundle = raw_bundle(tmp_path)
    first = certify_capture(tmp_path, bundle)
    feature_path = tmp_path / first["features"]["path"]
    original = feature_path.read_bytes()
    proof_path = tmp_path / first["feature_manifest"]["path"]
    assert certify_capture(tmp_path, bundle) == first
    assert len(registry(tmp_path)["races"]) == 1
    assert feature_path.read_bytes() == original
    feature_path.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="immutable"):
        certify_capture(tmp_path, bundle)
    assert feature_path.read_bytes() == original + b"tampered"
    assert file_sha256(proof_path) == first["feature_manifest"]["sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "constructor_missing",
        "wrong_event",
        "missing_schedule_role",
        "duplicate_qualifying_role",
        "wrong_url",
        "race_window_mismatch",
    ],
)
def test_invalid_frozen_inputs_are_never_registered_as_gold(tmp_path: Path, change: str) -> None:
    bundle = raw_bundle(tmp_path, change=change)
    with pytest.raises(ValueError):
        certify_capture(tmp_path, bundle)
    assert not (tmp_path / "data/benchmarks/prospective_registry.json").exists()
    assert not list((tmp_path / "data/features").rglob("*.parquet"))
    verify_bundle(bundle)


def test_tampered_raw_capture_is_rejected_before_normalization(tmp_path: Path) -> None:
    bundle = raw_bundle(tmp_path)
    (bundle / "qualifying.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        certify_capture(tmp_path, bundle)
    assert not (tmp_path / "data/benchmarks/prospective_registry.json").exists()


def audited_targets(root: Path) -> dict[str, str]:
    rows = [
        {
            "season": 2026,
            "round": 16,
            "driver_id": driver,
            "position": position,
            "classified": True,
            "winner": position == 1,
            "podium": True,
            "dnf": False,
            "dnf_category": "finished",
            "raw_status": "Finished",
            "taxonomy_version": DNF_TAXONOMY_VERSION,
            "final_audited": True,
            "label_available_at": CAPTURED + timedelta(days=2),
            "audit_reference": "fia:final-v1",
        }
        for position, driver in enumerate(("driver_a", "driver_b"), start=1)
    ]
    path = root / "targets.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=OUTCOME_SCHEMA), path)
    return {"path": path.name, "sha256": file_sha256(path)}


def test_independent_labels_join_each_capture_version_without_changing_features(
    tmp_path: Path,
) -> None:
    bundles = [raw_bundle(tmp_path), raw_bundle(tmp_path, captured=CAPTURED + timedelta(minutes=1))]
    certified = [certify_capture(tmp_path, bundle) for bundle in bundles]
    originals = {
        item["features"]["path"]: (tmp_path / item["features"]["path"]).read_bytes()
        for item in certified
    }
    hashes = [verify_bundle(bundle)["manifest_sha256"] for bundle in bundles]
    targets = audited_targets(tmp_path)
    result = attach_audited_outcomes(tmp_path, hashes, targets)
    assert result["eligible_capture_manifest_sha256"] == sorted(hashes)
    assert result["included_races"] == 2
    gold = pq.ParquetFile(tmp_path / "data/benchmarks/prospective/gold.parquet").read()
    assert gold.num_rows == 4
    assert sum(gold["label_winner"].to_pylist()) == 2
    assert (
        "label_winner"
        not in pq.ParquetFile(tmp_path / certified[0]["features"]["path"]).read().column_names
    )
    for entry in registry(tmp_path)["races"]:
        assert entry["outcomes"] == targets
        assert entry["evaluation_eligible"] is True
        assert (tmp_path / entry["features"]["path"]).read_bytes() == originals[
            entry["features"]["path"]
        ]
    benchmark_manifest = json.loads(
        (tmp_path / "data/benchmarks/prospective/manifest.json").read_text()
    )
    assert benchmark_manifest["catalog_sha256"] == file_sha256(
        tmp_path / "data/benchmarks/prospective_registry.json"
    )
    first_registry = (tmp_path / "data/benchmarks/prospective_registry.json").read_bytes()
    assert attach_audited_outcomes(tmp_path, hashes, targets) == result
    assert (tmp_path / "data/benchmarks/prospective_registry.json").read_bytes() == first_registry
    assert certify_capture(tmp_path, bundles[0])["evaluation_eligible"] is True


def test_target_hash_and_incomplete_roster_are_rejected(tmp_path: Path) -> None:
    bundle = raw_bundle(tmp_path)
    certify_capture(tmp_path, bundle)
    manifest_hash = verify_bundle(bundle)["manifest_sha256"]
    reference = audited_targets(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        attach_audited_outcomes(tmp_path, [manifest_hash], {**reference, "sha256": "0" * 64})
    path = tmp_path / reference["path"]
    pq.write_table(pq.ParquetFile(path).read().slice(0, 1), path)
    reference["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="field roster"):
        attach_audited_outcomes(tmp_path, [manifest_hash], reference)
    assert registry(tmp_path)["races"][0]["outcomes"] is None
