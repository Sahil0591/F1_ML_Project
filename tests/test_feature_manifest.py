import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.__main__ import main
from f1_ml_predictor.features.manifest import load_feature_request
from f1_ml_predictor.features.snapshot import build_snapshot


@pytest.fixture
def request_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "inputs.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "season=2025/round=02",
                    "driver_id": "max_verstappen",
                    "constructor_id": "red_bull",
                    "position": 1,
                    "q3_seconds": 90.0,
                }
            ]
        ),
        path,
    )
    common = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "available_at": "2025-03-15T09:30:00+00:00",
        "evidence_reference": "test-only-proof",
    }
    manifest = {
        "version": 1,
        "prediction_timestamp": "2025-03-15T10:00:00+00:00",
        "event": {
            "season": 2025,
            "round": 2,
            "circuit_id": "albert_park",
            "race_start": "2025-03-16T04:00:00+00:00",
            "qualifying_completed_at": "2025-03-15T09:00:00+00:00",
            "available_at": "2025-03-15T08:00:00+00:00",
            "evidence_reference": "test-only-schedule",
        },
        "rosters": [{**common, "kind": "roster"}],
        "qualifying": [{**common, "kind": "qualifying"}],
    }
    output = tmp_path / "manifest.json"
    output.write_text(json.dumps(manifest), encoding="utf-8")
    return output


def test_manifest_loads_hash_bound_inputs(request_manifest: Path) -> None:
    inputs, cutoff = load_feature_request(request_manifest, request_manifest.parent)
    assert build_snapshot(inputs, cutoff).num_rows == 1


@pytest.mark.parametrize("change", ["hash", "path", "kind", "timestamp"])
def test_manifest_rejects_tampering_and_invalid_contracts(
    request_manifest: Path, change: str
) -> None:
    payload = json.loads(request_manifest.read_text(encoding="utf-8"))
    item = payload["qualifying"][0]
    if change == "hash":
        item["sha256"] = "0" * 64
    elif change == "path":
        item["path"] = "../outside.parquet"
    elif change == "kind":
        item["kind"] = "race_results"
    else:
        item["available_at"] = "2025-03-15T09:30:00"
    request_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_feature_request(request_manifest, request_manifest.parent)


def test_feature_cli_builds_snapshot_offline(request_manifest: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "f1_ml_predictor",
            "build-snapshot",
            str(request_manifest),
            "--root",
            str(request_manifest.parent),
        ],
    )
    main()
    assert "rows: 1" in capsys.readouterr().out
    assert len(list((request_manifest.parent / "data" / "features").rglob("*.parquet"))) == 1
