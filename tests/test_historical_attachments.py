import json
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_historical import core_request
from test_trust_features import CUTOFF

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.trust import historical
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION, OUTCOME_SCHEMA


def test_delayed_target_attachment_preserves_frozen_features_and_joins_gold(tmp_path, monkeypatch):
    request_path, request = core_request(tmp_path, monkeypatch)
    result = historical.reconstruct_gold_core(request_path, tmp_path)
    manifest = tmp_path / result["evidence_manifest"]
    original_manifest = manifest.read_bytes()
    feature = tmp_path / result["features"]["path"]
    original_features = feature.read_bytes()
    labels = []
    for driver, position in [("driver_a", 1), ("driver_b", 2)]:
        labels.append(
            {
                "season": 2025,
                "round": 3,
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
                "label_available_at": CUTOFF + timedelta(hours=5),
                "audit_reference": "fixture final classification",
            }
        )
    outcome = tmp_path / "labels.parquet"
    pq.write_table(pa.Table.from_pylist(labels, schema=OUTCOME_SCHEMA), outcome)
    request["outcomes"] = {"path": outcome.name, "sha256": file_sha256(outcome)}
    binding = request["document_bindings"][0]
    request["outcome_document_bindings"] = [
        {
            "path": binding["path"],
            "sha256": binding["sha256"],
            "document_id": "23",
            "document_url": binding["document_url"],
            "event_id": binding["event_id"],
            "status": "final",
            "version_audited": True,
            "latest_final_audited": True,
            "audit_reference": "fixture final classification",
            "outcome_sha256": file_sha256(outcome),
            "registry_path": binding["registry_path"],
            "registry_sha256": binding["registry_sha256"],
            "label_available_at_utc": (CUTOFF + timedelta(hours=5)).isoformat(),
        }
    ]
    request_path.write_text(json.dumps(request))
    joined = historical.reconstruct_gold_core(request_path, tmp_path)
    assert joined["benchmark"]["datasets"]["Gold"]["rows"] == 2
    assert joined["benchmark"]["included_races"] == 1
    assert manifest.read_bytes() == original_manifest
    assert feature.read_bytes() == original_features
    assert (
        historical.reconstruct_gold_core(request_path, tmp_path)["snapshot_id"]
        == result["snapshot_id"]
    )


def test_corrupted_cached_feature_contents_are_not_adopted_as_gold(tmp_path, monkeypatch):
    path, _ = core_request(tmp_path, monkeypatch)
    result = historical.reconstruct_gold_core(path, tmp_path)
    feature = tmp_path / result["features"]["path"]
    stored = pq.ParquetFile(feature).read()
    rows = stored.to_pylist()
    rows[0]["qualifying_position"] = 99.0
    pq.write_table(pa.Table.from_pylist(rows, schema=stored.schema), feature)
    with pytest.raises(ValueError, match="persisted snapshot contents"):
        historical.reconstruct_gold_core(path, tmp_path)


def test_withdrawal_records_reason_without_changing_frozen_snapshot(tmp_path, monkeypatch):
    path, _ = core_request(tmp_path, monkeypatch)
    result = historical.reconstruct_gold_core(path, tmp_path)
    feature = tmp_path / result["features"]["path"]
    original = feature.read_bytes()
    report = historical.withdraw_gold_core(tmp_path, result["snapshot_id"], "fixture failed review")
    assert report["benchmark"]["included_races"] == 0
    assert feature.read_bytes() == original
    registry = json.loads((tmp_path / "data/benchmarks/gold_core_registry.json").read_text())
    assert registry["races"] == []
    assert registry["withdrawn_races"][0]["reason"] == "fixture failed review"
    with pytest.raises(ValueError, match="exact registered"):
        historical.withdraw_gold_core(tmp_path, result["snapshot_id"], "duplicate")
