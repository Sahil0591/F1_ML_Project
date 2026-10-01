"""A model can keep using earlier Gold eligibility and exact data bytes."""

import hashlib
import json

import pytest

from f1_ml_predictor.benchmarks import versioning
from f1_ml_predictor.benchmarks.versioning import archive_benchmark


def _write_benchmark(directory, status: str, payload: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "gold.parquet").write_bytes(payload)
    coverage = {
        "coverage": [
            {
                "event_id": "season=2026/round=01",
                "cutoff_kind": "post_qualifying",
                "status": status,
                "tier": "Gold" if status == "included" else None,
                "source_files": {"features": {"sha256": hashlib.sha256(payload).hexdigest()}},
            }
        ]
    }
    coverage_bytes = json.dumps(coverage).encode()
    (directory / "coverage.json").write_bytes(coverage_bytes)
    manifest = {
        "coverage_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "datasets": {
            "Gold": {
                "path": "gold.parquet",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "events": ["season=2026/round=01"] if status == "included" else [],
                "rows": 1 if status == "included" else 0,
            }
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_excluded_to_gold_preserves_both_immutable_benchmarks(tmp_path):
    _write_benchmark(tmp_path, "excluded", b"old")
    first = archive_benchmark(tmp_path)
    assert first is not None
    _write_benchmark(tmp_path, "included", b"new")
    second = archive_benchmark(tmp_path)
    assert second is not None
    assert first["dataset_version"] != second["dataset_version"]
    assert (first["path"] / "gold.parquet").read_bytes() == b"old"
    assert (second["path"] / "gold.parquet").read_bytes() == b"new"
    history = json.loads((second["path"] / "race_status.json").read_text())
    assert history["previous_version"] == first["dataset_version"]
    assert history["races"][0]["transition"] == "excluded_to_gold"
    assert history["races"][0]["previous"]["status"] == "excluded"
    assert archive_benchmark(tmp_path) == second
    transitions = (tmp_path / "versions" / "history.jsonl").read_text().splitlines()
    assert len(transitions) == 2
    assert json.loads(transitions[-1])["previous_version"] == first["dataset_version"]


def test_archiving_rejects_changed_data_under_existing_manifest(tmp_path):
    _write_benchmark(tmp_path, "included", b"original")
    first = archive_benchmark(tmp_path)
    assert first is not None
    (tmp_path / "gold.parquet").write_bytes(b"revised")
    try:
        archive_benchmark(tmp_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("changed benchmark bytes were accepted")


def test_new_gold_event_records_prior_absence_as_excluded(tmp_path):
    _write_benchmark(tmp_path, "excluded", b"old")
    empty = json.dumps({"coverage": []}).encode()
    (tmp_path / "coverage.json").write_bytes(empty)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["coverage_sha256"] = hashlib.sha256(empty).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    first = archive_benchmark(tmp_path)
    assert first is not None
    _write_benchmark(tmp_path, "included", b"new")
    second = archive_benchmark(tmp_path)
    assert second is not None
    history = json.loads((second["path"] / "race_status.json").read_text())
    assert history["races"][0]["transition"] == "excluded_to_gold"
    assert history["races"][0]["previous"]["reasons"] == ["absent_from_previous_benchmark"]


def test_failed_copy_does_not_publish_partial_version(tmp_path, monkeypatch):
    _write_benchmark(tmp_path, "included", b"gold")
    original_copy = versioning.shutil.copyfile

    def interrupted_copy(source, destination):
        if source.name == "coverage.json":
            raise OSError("interrupted copy")
        return original_copy(source, destination)

    monkeypatch.setattr(versioning.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        archive_benchmark(tmp_path)
    assert not list((tmp_path / "versions").glob("dataset-*"))
    assert not (tmp_path / "versions" / "current.json").exists()

    monkeypatch.setattr(versioning.shutil, "copyfile", original_copy)
    assert archive_benchmark(tmp_path) is not None


def test_existing_version_rejects_changed_race_status(tmp_path):
    _write_benchmark(tmp_path, "included", b"gold")
    snapshot = archive_benchmark(tmp_path)
    assert snapshot is not None
    status_path = snapshot["path"] / "race_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["manifest_sha256"] = "invalid"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match="race status mismatch"):
        archive_benchmark(tmp_path)
