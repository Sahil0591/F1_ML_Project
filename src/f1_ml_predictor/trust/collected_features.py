"""Certify lean features from immutable live bundles without acquiring later data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import build_benchmarks
from f1_ml_predictor.features.contracts import FeatureInputs, PreRaceEvent, PublishedTable
from f1_ml_predictor.features.snapshot import build_snapshot
from f1_ml_predictor.features.storage import persist_snapshot
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import (
    ENTRY_SCHEMA,
    normalize_qualifying,
    normalize_schedule,
)
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.sources.jolpica import BASE_URL
from f1_ml_predictor.trust.cutoffs import CutoffKind
from f1_ml_predictor.trust.evidence import AvailabilityEvidence, EvidenceClass, table_hash
from f1_ml_predictor.trust.locking import advisory_lock
from f1_ml_predictor.trust.outcomes import OUTCOME_SCHEMA, validate_audited_outcomes
from f1_ml_predictor.trust.prospective import load_bundle

_REGISTRY = Path("data/benchmarks/prospective_registry.json")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    member = Path(relative)
    path = (root / member).resolve()
    if member.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("prospective artifact must remain within the project root")
    return path


def _publish(path: Path, data: bytes) -> None:
    """Publish an immutable artifact or verify an identical retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError("immutable prospective artifact changed") from None


def _table_reference(root: Path, relative: Path, table: pa.Table) -> dict[str, str]:
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    data = buffer.getvalue().to_pybytes()
    _publish(_inside(root, relative.as_posix()), data)
    return {"path": relative.as_posix(), "sha256": _digest(data)}


def _snapshot_reference(root: Path, table: pa.Table) -> dict[str, str]:
    """Stage through the existing writer before publishing without overwriting."""
    directory = root / "data/features"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prospective-", dir=directory) as temporary:
        staged_root = Path(temporary)
        staged = persist_snapshot(StoragePaths(staged_root), table)
        loaded = pq.ParquetFile(staged).read().replace_schema_metadata(None)
        if not loaded.equals(table.replace_schema_metadata(None)):
            raise ValueError("persisted prospective snapshot differs from its certified table")
        relative = staged.relative_to(staged_root)
        data = staged.read_bytes()
        destination = _inside(root, relative.as_posix())
        _publish(destination, data)
    actual = pq.ParquetFile(destination).read().replace_schema_metadata(None)
    if not actual.equals(table.replace_schema_metadata(None)):
        raise ValueError("prospective snapshot verification failed")
    return {"path": relative.as_posix(), "sha256": _digest(data)}


@contextmanager
def _registry_lock(root: Path) -> Iterator[None]:
    with advisory_lock(root / "data/benchmarks/.prospective_registry.lock"):
        yield


def _read_registry(root: Path) -> dict[str, Any]:
    path = root / _REGISTRY
    if not path.exists():
        return {"version": 1, "races": []}
    registry = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(registry, dict)
        or type(registry.get("version")) is not int
        or registry["version"] != 1
        or not isinstance(registry.get("races"), list)
        or not all(isinstance(row, dict) for row in registry["races"])
    ):
        raise ValueError("invalid prospective registry")
    return dict(registry)


def _save_registry(root: Path, registry: dict[str, Any]) -> None:
    path = root / _REGISTRY
    descriptor, name = tempfile.mkstemp(prefix=".prospective-registry-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(registry))
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _collection(payload: Any, *, qualifying: bool) -> list[dict[str, Any]]:
    try:
        mrdata = payload["MRData"]
        races = mrdata["RaceTable"]["Races"]
        raw_total, raw_limit, raw_offset = (mrdata[key] for key in ("total", "limit", "offset"))
        if any(
            isinstance(value, bool)
            or not isinstance(value, (str, int))
            or not str(value).isdecimal()
            for value in (raw_total, raw_limit, raw_offset)
        ):
            raise ValueError("invalid retained Jolpica pagination")
        total, limit, offset = int(raw_total), int(raw_limit), int(raw_offset)
        if (
            not isinstance(races, list)
            or not all(isinstance(row, dict) for row in races)
            or offset != 0
            or not 1 <= limit <= 100
            or not 0 <= total <= limit
        ):
            raise ValueError("retained Jolpica collection is incomplete")
        count = (
            sum(len(row.get("QualifyingResults", [])) for row in races)
            if qualifying
            else len(races)
        )
        if count != total or not total or (qualifying and len(races) != 1):
            raise ValueError("retained Jolpica collection is incomplete")
        return races
    except (KeyError, TypeError) as exc:
        raise ValueError("retained Jolpica payload is malformed") from exc


def _evidence(table: pa.Table, captured: datetime, reference: str) -> AvailabilityEvidence:
    return AvailabilityEvidence(
        EvidenceClass.CAPTURED_LIVE, reference, captured, captured, table_hash(table)
    )


def certify_capture(root: Path, bundle: Path) -> dict[str, Any]:
    """Build and register certified features exclusively from a verified raw bundle.

    Grid, history, practice, weather, and standings remain missing in this lean
    cohort. The qualifying-derived roster records exactly the retained driver IDs.
    Certification does not imply that later audited outcomes are available.
    """
    root, bundle = root.resolve(), bundle.resolve()
    if not bundle.is_relative_to(root):
        raise ValueError("prospective capture must remain within the project root")
    manifest, tables = load_bundle(bundle)
    identity = EventId(**manifest["event"])
    captured = datetime.fromisoformat(manifest["captured_at"])
    cutoff = datetime.fromisoformat(manifest["cutoff"])
    manifest_hash = manifest["manifest_sha256"]
    requests = manifest["request_metadata"].get("source_requests")
    window = manifest["request_metadata"].get("window")
    if not isinstance(requests, dict) or not isinstance(window, dict):
        raise ValueError("capture requires frozen source requests and prediction window")
    sources: dict[str, dict[str, Any]] = {}
    payloads: dict[str, Any] = {}
    for name, request in requests.items():
        if not isinstance(request, dict) or name not in tables:
            raise ValueError("capture source request is not bound to a raw input")
        table = tables[name]
        if table.column_names != ["payload_json"] or table.num_rows != 1:
            raise ValueError("capture inputs must contain one retained JSON payload")
        raw = table["payload_json"][0].as_py()
        if not isinstance(raw, str):
            raise ValueError("capture payload must be retained JSON text")
        payload = json.loads(raw)
        payloads[name] = payload
        sources[name] = {
            **request,
            "payload_sha256": _digest(raw.encode()),
            "canonical_payload_sha256": _digest(_canonical(payload)),
            "raw_table_hash": manifest["inputs"][name]["table_hash"],
        }
    role_names = {
        role: [name for name, request in requests.items() if request.get("role") == role]
        for role in ("event", "qualifying")
    }
    if any(len(names) != 1 for names in role_names.values()):
        raise ValueError("certification requires exactly one schedule and qualifying input")
    schedule_name, qualifying_name = role_names["event"][0], role_names["qualifying"][0]
    expected = {
        schedule_name: f"{BASE_URL}/{identity.season}",
        qualifying_name: f"{BASE_URL}/{identity.season}/{identity.round}/qualifying",
    }
    if any(requests[name].get("url", "").rstrip("/") != url for name, url in expected.items()):
        raise ValueError(
            "captured schedule and qualifying must use the exact Jolpica event endpoints"
        )
    schedule = normalize_schedule(
        _collection(payloads[schedule_name], qualifying=False), identity.season
    )
    selected = [row for row in schedule.to_pylist() if row["event_id"] == identity.partition()]
    if len(selected) != 1 or selected[0]["race_start_utc"] is None:
        raise ValueError("captured schedule is missing the exact race start")
    race_start = selected[0]["race_start_utc"]
    if race_start != datetime.fromisoformat(window["race_start"]):
        raise ValueError("frozen schedule and prediction window race starts disagree")
    decision = datetime.fromisoformat(window["qualifying_decision_at"])
    qualifying = normalize_qualifying(
        _collection(payloads[qualifying_name], qualifying=True), identity
    )
    q_rows = [{**row, "available_at": captured} for row in qualifying.to_pylist()]
    qualifying = pa.Table.from_pylist(q_rows, schema=qualifying.schema)
    roster = pa.Table.from_pylist(
        [{key: row[key] for key in ENTRY_SCHEMA.names} for row in q_rows], schema=ENTRY_SCHEMA
    )

    def reference(source_name: str, derived: str) -> str:
        return (
            f"prospective:manifest_sha256={manifest_hash};source={source_name};"
            f"payload_sha256={sources[source_name]['payload_sha256']};derived={derived}"
        )

    event_reference = reference(schedule_name, "event-v1")
    event = PreRaceEvent(
        identity, selected[0]["circuit_id"], race_start, decision, captured, event_reference
    )
    event = replace(event, evidence=_evidence(event.as_table(), captured, event_reference))
    roster_reference, q_reference = (
        reference(qualifying_name, "roster-v1"),
        reference(qualifying_name, "qualifying-v1"),
    )
    roster_proof, q_proof = (
        _evidence(roster, captured, roster_reference),
        _evidence(qualifying, captured, q_reference),
    )
    inputs = FeatureInputs(
        event,
        (PublishedTable(roster, captured, roster_reference, roster_proof),),
        (PublishedTable(qualifying, captured, q_reference, q_proof),),
    )
    kind = CutoffKind(manifest["cutoff_kind"])
    minutes = window.get("pre_race_minutes", 60)
    snapshot = build_snapshot(
        inputs, cutoff, cutoff_kind=kind, pre_race_minutes=minutes, certified_only=True
    )
    if set(snapshot["benchmark_tier"].to_pylist()) != {"Gold"}:
        raise ValueError("normalized capture did not produce a certified Gold snapshot")
    derived = Path("data/features/prospective_inputs") / manifest_hash
    roster_file = _table_reference(root, derived / "roster.parquet", roster)
    qualifying_file = _table_reference(root, derived / "qualifying.parquet", qualifying)
    snapshot_file = _snapshot_reference(root, snapshot)
    feature_request = {
        "version": 2,
        "prediction_timestamp": cutoff.isoformat(),
        "cutoff_kind": kind.value,
        "pre_race_minutes": minutes,
        "certified_only": True,
        "capture_bundle": bundle.relative_to(root).as_posix(),
        "capture_manifest_sha256": manifest_hash,
        "raw_sources": sources,
        "event": {
            "season": identity.season,
            "round": identity.round,
            "circuit_id": event.circuit_id,
            "race_start": race_start.isoformat(),
            "qualifying_completed_at": decision.isoformat(),
            "qualifying_status": "completed",
            "available_at": captured.isoformat(),
            "evidence_reference": event_reference,
            "evidence": event.evidence.to_dict() if event.evidence else None,
        },
        "rosters": [
            {
                **roster_file,
                "kind": "roster",
                "available_at": captured.isoformat(),
                "evidence_reference": roster_reference,
                "evidence": roster_proof.to_dict(),
            }
        ],
        "qualifying": [
            {
                **qualifying_file,
                "kind": "qualifying",
                "available_at": captured.isoformat(),
                "evidence_reference": q_reference,
                "evidence": q_proof.to_dict(),
            }
        ],
        "feature_snapshot": snapshot_file,
        "unused_raw_inputs": sorted(set(requests) - {schedule_name, qualifying_name}),
    }
    proof_path = Path("data/features/prospective_evidence") / f"{manifest_hash}.json"
    proof_bytes = _canonical(feature_request)
    _publish(_inside(root, proof_path.as_posix()), proof_bytes)
    entry = {
        "event_id": identity.partition(),
        "prediction_timestamp": cutoff.isoformat(),
        "cutoff_kind": kind.value,
        "features": snapshot_file,
        "outcomes": None,
        "capture_manifest_sha256": manifest_hash,
        "feature_manifest": {"path": proof_path.as_posix(), "sha256": _digest(proof_bytes)},
        "benchmark_tier": "Gold",
        "evaluation_eligible": False,
    }
    with _registry_lock(root):
        registry = _read_registry(root)
        existing = [
            row for row in registry["races"] if row.get("capture_manifest_sha256") == manifest_hash
        ]
        if existing:
            if len(existing) != 1 or any(
                existing[0].get(key) != entry[key]
                for key in (
                    "event_id",
                    "prediction_timestamp",
                    "cutoff_kind",
                    "features",
                    "feature_manifest",
                )
            ):
                raise ValueError("immutable registered feature capture changed")
            entry = existing[0]
        else:
            registry["races"].append(entry)
            _save_registry(root, registry)
    return {
        "features": snapshot_file,
        "feature_manifest": entry["feature_manifest"],
        "registry": _REGISTRY.as_posix(),
        "benchmark_tier": "Gold",
        "gold_snapshot": True,
        "normalization_required": False,
        "evaluation_eligible": entry["evaluation_eligible"],
    }


def attach_audited_outcomes(
    root: Path, capture_manifest_hashes: list[str], outcome_reference: dict[str, str]
) -> dict[str, Any]:
    """Attach independent audited targets and build the registered prospective cohorts."""
    root = root.resolve()
    outcome_path = _inside(root, outcome_reference["path"])
    data = outcome_path.read_bytes()
    if _digest(data) != outcome_reference["sha256"]:
        raise ValueError("registered audited outcome hash mismatch")
    outcomes = pq.ParquetFile(pa.BufferReader(data)).read()
    if not outcomes.schema.remove_metadata().equals(OUTCOME_SCHEMA):
        raise ValueError("registered outcomes require the exact audited schema")
    with _registry_lock(root):
        registry = _read_registry(root)
        selected = [
            row
            for row in registry["races"]
            if row.get("capture_manifest_sha256") in capture_manifest_hashes
        ]
        if len(selected) != len(set(capture_manifest_hashes)):
            raise ValueError(
                "outcome import requires all certified capture versions in the registry"
            )
        for entry in selected:
            feature_path = _inside(root, entry["features"]["path"])
            feature_bytes = feature_path.read_bytes()
            if _digest(feature_bytes) != entry["features"]["sha256"]:
                raise ValueError("registered feature snapshot hash changed")
            features = pq.ParquetFile(pa.BufferReader(feature_bytes)).read()
            rows = features.to_pylist()
            event_text = entry["event_id"]
            season, round_ = (int(part.split("=")[1]) for part in event_text.split("/"))
            event = EventId(season, round_)
            validate_audited_outcomes(
                outcomes, field_roster={(event, row["driver_id"]) for row in rows}
            )
            if sum(bool(row["winner"]) for row in outcomes.to_pylist()) != 1:
                raise ValueError("complete audited outcome import requires exactly one winner")
        for entry in selected:
            previous = entry.get("outcomes")
            history = entry.setdefault("outcome_versions", [])
            if previous is not None and previous not in history:
                history.append(previous)
            if outcome_reference not in history:
                history.append(outcome_reference)
            entry["outcomes"] = dict(outcome_reference)
            entry["evaluation_eligible"] = True
        _save_registry(root, registry)
        report = build_benchmarks(root, root / "data/benchmarks/prospective", root / _REGISTRY)
        eligible = []
        changed = False
        for entry in selected:
            matching = [
                row
                for row in report["coverage"]
                if (
                    row["event_id"] == entry["event_id"]
                    and row.get("cutoff_kind") == entry["cutoff_kind"]
                    and row.get("source_files", {}).get("features") == entry["features"]
                )
            ]
            if (
                len(matching) == 1
                and matching[0]["status"] == "included"
                and matching[0].get("tier") == "Gold"
            ):
                eligible.append(entry["capture_manifest_sha256"])
            else:
                entry["evaluation_eligible"] = False
                changed = True
        if changed:
            _save_registry(root, registry)
            report = build_benchmarks(root, root / "data/benchmarks/prospective", root / _REGISTRY)
        return {
            "registry": _REGISTRY.as_posix(),
            "eligible_capture_manifest_sha256": sorted(eligible),
            "benchmark": "data/benchmarks/prospective",
            "included_races": report["included_races"],
        }
