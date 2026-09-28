"""Offline checks for immutable prospective captures."""

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust import prospective

CAPTURE = datetime(2026, 9, 28, 12, tzinfo=UTC)
EVENT = EventId(2026, 18)


def freeze(root, **kwargs):
    values = dict(
        root=root,
        event=EVENT,
        cutoff=CAPTURE + timedelta(minutes=1),
        cutoff_kind="pre_race",
        captured_at=CAPTURE,
        tables={"qualifying": pa.table({"driver_id": ["norris"], "position": [1]})},
        request_metadata={"source": "live"},
        now=CAPTURE,
    )
    values.update(kwargs)
    return prospective.freeze_bundle(**values)


def rewrite_manifest(path, mutate, rehash=False):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mutate(manifest)
    if rehash:
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        manifest["manifest_sha256"] = hashlib.sha256(prospective._canonical(body)).hexdigest()
    manifest_path.write_bytes(prospective._canonical(manifest))


def test_round_trip_and_identical_retry(tmp_path):
    path = freeze(tmp_path)
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    assert freeze(tmp_path) == path
    assert {file.name: file.read_bytes() for file in path.iterdir()} == before
    manifest, tables = prospective.load_bundle(path)
    assert tables["qualifying"].to_pylist() == [{"driver_id": "norris", "position": 1}]
    evidence = manifest["inputs"]["qualifying"]["evidence"]
    assert evidence["class"] == "captured_live"
    assert evidence["available_at"] == evidence["captured_at"] == CAPTURE.isoformat()
    assert prospective.verify_bundle(path) == manifest


def test_identical_retry_can_reuse_a_historical_verified_capture(tmp_path):
    path = freeze(tmp_path)
    assert freeze(tmp_path, now=CAPTURE + timedelta(days=100)) == path


def test_old_capture_cannot_be_published_to_a_new_slot(tmp_path):
    with pytest.raises(ValueError, match="fresh"):
        freeze(tmp_path, now=CAPTURE + timedelta(days=100))
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob(".pending-*"))


@pytest.mark.parametrize(
    "changes",
    [
        {"tables": {"qualifying": pa.table({"position": [2]})}},
        {"request_metadata": {"source": "different"}},
        {"captured_at": CAPTURE - timedelta(seconds=1)},
    ],
)
def test_slot_collision_never_overwrites(tmp_path, changes):
    path = freeze(tmp_path)
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="different capture"):
        freeze(tmp_path, **changes)
    assert (path / "manifest.json").read_bytes() == before
    assert not list(path.parent.glob(".pending-*"))


def test_empty_table_is_a_valid_capture(tmp_path):
    path = freeze(
        tmp_path, tables={"qualifying": pa.table({"position": pa.array([], type=pa.int64())})}
    )
    assert prospective.load_bundle(path)[1]["qualifying"].num_rows == 0


def test_interruption_leaves_no_published_or_pending_bundle(tmp_path, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(prospective.pq, "write_table", interrupt)
    with pytest.raises(KeyboardInterrupt):
        freeze(tmp_path)
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob(".pending-*"))


def test_atomic_rename_failure_cleans_pending_bundle(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("interrupted publication")

    monkeypatch.setattr(prospective.Path, "rename", fail)
    with pytest.raises(OSError, match="interrupted"):
        freeze(tmp_path)
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob(".pending-*"))


def test_corrupt_file_rejects_load_and_retry(tmp_path):
    path = freeze(tmp_path)
    (path / "qualifying.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        prospective.load_bundle(path)
    with pytest.raises(ValueError, match="checksum"):
        freeze(tmp_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(captured_at=(CAPTURE - timedelta(days=1)).isoformat()),
        lambda m: m["request_metadata"].update(source="altered"),
        lambda m: m["inputs"]["qualifying"].update(table_hash="0" * 64),
    ],
)
def test_metadata_checksum_binds_capture_and_requests(tmp_path, mutate):
    path = freeze(tmp_path)
    rewrite_manifest(path, mutate)
    with pytest.raises(ValueError, match="checksum"):
        prospective.verify_bundle(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["inputs"]["qualifying"].update(file="../outside.parquet"),
        lambda m: m["inputs"]["qualifying"]["evidence"].update(
            captured_at=(CAPTURE - timedelta(days=1)).isoformat()
        ),
        lambda m: m.update(cutoff_kind="post_qualifying"),
        lambda m: m["inputs"]["qualifying"].update(rows=10),
    ],
)
def test_structural_integrity_even_with_updated_manifest_hash(tmp_path, mutate):
    path = freeze(tmp_path)
    rewrite_manifest(path, mutate, rehash=True)
    with pytest.raises(ValueError):
        prospective.verify_bundle(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"captured_at": CAPTURE.replace(tzinfo=None)},
        {"cutoff": CAPTURE.replace(tzinfo=timezone(timedelta(hours=1)))},
        {"captured_at": CAPTURE + timedelta(seconds=1)},
        {"captured_at": CAPTURE - timedelta(days=1)},
        {"cutoff": CAPTURE - timedelta(seconds=1)},
        {"cutoff_kind": "after_race"},
        {"now": CAPTURE.replace(tzinfo=None)},
    ],
)
def test_invalid_timestamps_and_kinds(tmp_path, changes):
    with pytest.raises(ValueError):
        freeze(tmp_path, **changes)
    assert not list(tmp_path.rglob("manifest.json"))


@pytest.mark.parametrize(
    "tables",
    [
        {"../escape": pa.table({"position": [1]})},
        {"labels": pa.table({"position": [1]})},
        {"qualifying": pa.table({"target_position": [1]})},
        {},
    ],
)
def test_reject_labels_and_unsafe_names(tmp_path, tables):
    with pytest.raises(ValueError):
        freeze(tmp_path, tables=tables)


def test_extra_file_is_not_a_complete_valid_bundle(tmp_path):
    path = freeze(tmp_path)
    (path / "labels.parquet").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        prospective.verify_bundle(path)
