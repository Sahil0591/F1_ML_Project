"""Immutable, locally captured predictive inputs.

Version 1 bundles live at ``root/season=N/round=NN/kind/UTC-cutoff``.
``manifest.json`` contains event/cutoff/capture metadata, request metadata, and
an ``inputs`` mapping with Parquet file SHA-256, logical table hash, row count,
and captured-live evidence. ``manifest_sha256`` hashes the canonical JSON of
all other manifest fields. Checksums detect corruption, not malicious authors
who can rewrite the entire bundle. Capture timestamps must come from the live
collector; new bundles accept only captures within five minutes of the clock.
Verification and identical retries do not apply this freshness restriction.
"""

import hashlib
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.evidence import (
    AvailabilityEvidence,
    EvidenceClass,
    table_hash,
)

_KINDS = {"post_qualifying", "provisional_grid", "pre_race"}
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_LABEL_NAMES = {"labels", "targets", "outcomes", "results", "race_results"}
_CAPTURE_MAX_AGE = timedelta(minutes=5)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _timestamp(value: datetime, name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be timezone-aware UTC")
    require_utc(value, name)


def _slot_name(cutoff: datetime) -> str:
    return cutoff.strftime("%Y%m%dT%H%M%S.%fZ")


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME.fullmatch(name) or name in _LABEL_NAMES:
        raise ValueError("input names must be safe predictive table identifiers")


def _validate_table(table: pa.Table) -> None:
    if not isinstance(table, pa.Table):
        raise ValueError("inputs must be Arrow tables")
    if any(
        name.lower() in {"label", "target"} or name.lower().startswith(("label_", "target_"))
        for name in table.column_names
    ):
        raise ValueError("labels cannot be included in predictive captures")


def freeze_bundle(
    root: Path,
    event: EventId,
    cutoff: datetime,
    cutoff_kind: str,
    captured_at: datetime,
    tables: dict[str, pa.Table],
    request_metadata: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> Path:
    """Publish a complete capture or reuse an identical immutable slot.

    ``now`` is a clock injection for offline tests. Production callers should
    omit it. Different capture metadata also constitutes a slot collision.
    """
    _timestamp(cutoff, "cutoff")
    _timestamp(captured_at, "captured_at")
    clock = datetime.now(UTC) if now is None else now
    _timestamp(clock, "now")
    if not isinstance(event, EventId) or cutoff_kind not in _KINDS:
        raise ValueError("invalid event or cutoff kind")
    require_known_by(captured_at, cutoff)
    if not isinstance(tables, dict) or not tables:
        raise ValueError("at least one predictive input table is required")
    if request_metadata is not None and not isinstance(request_metadata, dict):
        raise ValueError("request_metadata must be a JSON object")
    metadata = json.loads(_canonical(request_metadata or {}))
    for name, table in tables.items():
        _validate_name(name)
        _validate_table(table)
    root = Path(root).resolve()
    path = root / event.partition() / cutoff_kind / _slot_name(cutoff)
    if not path.resolve().is_relative_to(root):
        raise ValueError("bundle slot escapes root")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=path.parent))
    try:
        inputs: dict[str, Any] = {}
        for name, table in sorted(tables.items()):
            filename = f"{name}.parquet"
            pq.write_table(table, temporary / filename)
            digest = table_hash(table)
            inputs[name] = {
                "file": filename,
                "sha256": _sha256((temporary / filename).read_bytes()),
                "table_hash": digest,
                "rows": table.num_rows,
                "evidence": AvailabilityEvidence(
                    EvidenceClass.CAPTURED_LIVE,
                    filename,
                    captured_at,
                    captured_at,
                    digest,
                ).to_dict(),
            }
        manifest = {
            "format_version": 1,
            "event": {"season": event.season, "round": event.round},
            "cutoff_kind": cutoff_kind,
            "cutoff": cutoff.isoformat(),
            "captured_at": captured_at.isoformat(),
            "request_metadata": metadata,
            "inputs": inputs,
        }
        manifest["manifest_sha256"] = _sha256(_canonical(manifest))
        (temporary / "manifest.json").write_bytes(_canonical(manifest))
        if path.exists():
            existing = verify_bundle(path)
            if existing != manifest:
                raise ValueError("immutable bundle slot already contains a different capture")
            return path
        if captured_at > clock or clock - captured_at > _CAPTURE_MAX_AGE:
            raise ValueError("capture must be fresh and cannot be backdated or in the future")
        try:
            temporary.rename(path)
        except OSError:
            if not path.exists():
                raise
            existing = verify_bundle(path)
            if existing != manifest:
                raise ValueError(
                    "immutable bundle slot already contains a different capture"
                ) from None
        return path
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def load_bundle(path: Path) -> tuple[dict[str, Any], dict[str, pa.Table]]:
    """Read and validate a complete capture before returning any input tables."""
    path = Path(path)
    try:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("bundle must be a real directory")
        manifest_path = path / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("manifest cannot be a symlink")
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
        expected_keys = {
            "format_version",
            "event",
            "cutoff_kind",
            "cutoff",
            "captured_at",
            "request_metadata",
            "inputs",
            "manifest_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_keys:
            raise ValueError("invalid manifest fields")
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if raw != _canonical(manifest) or manifest["manifest_sha256"] != _sha256(_canonical(body)):
            raise ValueError("manifest checksum mismatch")
        if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
            raise ValueError("unsupported prospective bundle version")
        if not isinstance(manifest["event"], dict) or set(manifest["event"]) != {"season", "round"}:
            raise ValueError("invalid event metadata")
        event = EventId(**manifest["event"])
        cutoff = datetime.fromisoformat(manifest["cutoff"])
        captured_at = datetime.fromisoformat(manifest["captured_at"])
        require_known_by(captured_at, cutoff)
        kind = manifest["cutoff_kind"]
        if kind not in _KINDS or not isinstance(manifest["request_metadata"], dict):
            raise ValueError("invalid capture metadata")
        suffix = Path(event.partition()) / kind / _slot_name(cutoff)
        if tuple(path.parts[-len(suffix.parts) :]) != suffix.parts:
            raise ValueError("manifest does not match bundle slot")
        inputs = manifest["inputs"]
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("missing prospective inputs")
        tables: dict[str, pa.Table] = {}
        for name, item in inputs.items():
            _validate_name(name)
            if not isinstance(item, dict) or set(item) != {
                "file",
                "sha256",
                "table_hash",
                "rows",
                "evidence",
            }:
                raise ValueError("invalid input manifest")
            if item["file"] != f"{name}.parquet":
                raise ValueError("input filename is not a safe bundle member")
            file_path = path / item["file"]
            if file_path.is_symlink():
                raise ValueError("inputs cannot be symlinks")
            data = file_path.read_bytes()
            if _sha256(data) != item["sha256"]:
                raise ValueError("input file checksum mismatch")
            table = pq.read_table(pa.BufferReader(data))
            _validate_table(table)
            digest = table_hash(table)
            expected_evidence = AvailabilityEvidence(
                EvidenceClass.CAPTURED_LIVE, item["file"], captured_at, captured_at, digest
            ).to_dict()
            if (
                type(item["rows"]) is not int
                or item["rows"] != table.num_rows
                or item["table_hash"] != digest
                or item["evidence"] != expected_evidence
            ):
                raise ValueError("input table or capture evidence mismatch")
            tables[name] = table
        expected_files = {"manifest.json", *(f"{name}.parquet" for name in inputs)}
        if {member.name for member in path.iterdir()} != expected_files:
            raise ValueError("bundle contains unexpected files")
        return manifest, tables
    except (OSError, TypeError, KeyError, pa.ArrowException) as exc:
        raise ValueError("invalid or incomplete prospective bundle") from exc


def verify_bundle(path: Path) -> dict[str, Any]:
    """Return the manifest only after verifying every captured input."""
    return load_bundle(path)[0]
