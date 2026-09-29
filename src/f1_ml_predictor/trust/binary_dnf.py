"""Versioned binary DNF audit from final FIA, OpenF1 and Jolpica outcomes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file, build_benchmarks
from f1_ml_predictor.trust.outcomes import OUTCOME_SCHEMA, DnfCategory, validate_audited_outcomes

AUDIT_VERSION = "binary-dnf-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable DNF audit artifact changed: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(content)


def _payload(root: Path, record: dict[str, Any]) -> Any:
    return json.loads(_safe_file(root, record["path"], record["sha256"]).read_bytes())


def audit_binary_status(
    fia: dict[str, Any],
    jolpica: dict[str, Any] | None,
    openf1: dict[str, Any] | None,
) -> tuple[DnfCategory | None, str]:
    """Require independent status agreement; DNS and DSQ remain outside binary DNF."""
    existing = DnfCategory(fia["dnf_category"])
    if existing in {DnfCategory.DID_NOT_START, DnfCategory.DISQUALIFIED}:
        expected = "dns" if existing == DnfCategory.DID_NOT_START else "dsq"
        if jolpica is None or openf1 is None:
            return None, f"{expected}_source_missing"
        status = "Did not start" if expected == "dns" else "Disqualified"
        flags = (openf1.get("dnf"), openf1.get("dns"), openf1.get("dsq"))
        wanted = (False, True, False) if expected == "dns" else (False, False, True)
        return None, expected if jolpica.get(
            "status"
        ) == status and flags == wanted else f"{expected}_disagreement"
    if jolpica is None or openf1 is None:
        return None, "source_missing"
    flags = (openf1.get("dnf"), openf1.get("dns"), openf1.get("dsq"))
    if not all(type(flag) is bool for flag in flags) or sum(bool(flag) for flag in flags) > 1:
        return None, "invalid_openf1_flags"
    if flags[1] or flags[2]:
        return None, "openf1_dns_or_dsq_disagreement"
    source_status = jolpica.get("status")
    raw = fia["raw_status"].upper()
    if flags[0] and source_status == "Retired" and raw == "DNF":
        return DnfCategory.RETIRED_OTHER, "three_source_retired_agreement"
    if (
        not flags[0]
        and source_status in {"Finished", "Lapped"}
        and fia["classified"]
        and raw == "CLASSIFICATION_ONLY"
    ):
        return DnfCategory.FINISHED, "three_source_finished_agreement"
    return None, "status_disagreement_or_ambiguous_fia"


def build_binary_dnf_benchmark(root: Path, capture_path: Path) -> dict[str, Any]:
    """Freeze new outcome versions and a separate Gold benchmark catalog."""
    root = root.resolve()
    capture_bytes = capture_path.read_bytes()
    capture = json.loads(capture_bytes)
    capture_hash = hashlib.sha256(capture_bytes).hexdigest()
    registry_path = root / "data/benchmarks/gold_core_registry.json"
    registry_bytes = registry_path.read_bytes()
    if (
        capture["version"] != 1
        or capture["registry_sha256"] != hashlib.sha256(registry_bytes).hexdigest()
    ):
        raise ValueError("DNF capture does not match the current Gold registry")
    registry = json.loads(registry_bytes)
    captures = {item["event_id"]: item for item in capture["races"]}
    if set(captures) != {item["event_id"] for item in registry["races"]}:
        raise ValueError("DNF capture does not cover the complete Gold cohort")
    directory = root / "data/benchmarks/gold_core_binary_dnf_v1" / capture_hash
    audit: dict[str, Any] = {
        "version": AUDIT_VERSION,
        "source_gold_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "capture_sha256": capture_hash,
        "label_clock_policy": (
            "original audited FIA final-classification publication bound; current API records "
            "corroborate retrospective binary outcome and are not pre-race inputs"
        ),
        "races": [],
    }
    revised = json.loads(registry_bytes)
    for race in revised["races"]:
        event_id = race["event_id"]
        original = race["outcomes"]
        final = pq.read_table(_safe_file(root, original["path"], original["sha256"]))
        validate_audited_outcomes(final)
        capture_race = captures[event_id]
        sources = {}
        report: dict[str, Any] = {
            "event_id": event_id,
            "original_outcomes": original,
            "status": capture_race["status"],
            "rows": [],
        }
        if capture_race["status"] == "captured":
            jolpica = _payload(root, capture_race["jolpica"])["MRData"]["RaceTable"]["Races"]
            if len(jolpica) != 1 or (int(jolpica[0]["season"]), int(jolpica[0]["round"])) != (
                final["season"][0].as_py(),
                final["round"][0].as_py(),
            ):
                raise ValueError("DNF Jolpica race identity mismatch")
            results = jolpica[0]["Results"]
            by_driver = {item["Driver"]["driverId"]: item for item in results}
            by_number = {
                int(item["Driver"]["permanentNumber"]): item
                for item in results
                if item["Driver"].get("permanentNumber")
            }
            numbered = sum(bool(item["Driver"].get("permanentNumber")) for item in results)
            if len(by_driver) != len(results) or len(by_number) != numbered:
                raise ValueError("DNF Jolpica driver identities are duplicated")
            openf1_rows = _payload(root, capture_race["openf1_session_result"])
            by_openf1 = {item["driver_number"]: item for item in openf1_rows}
            if len(by_openf1) != len(openf1_rows) or any(
                item["session_key"] != capture_race["openf1_session_key"] for item in openf1_rows
            ):
                raise ValueError("DNF OpenF1 session result is duplicated or misbound")
            sources = {
                name: capture_race[name]
                for name in ("jolpica", "openf1_sessions", "openf1_session_result")
            }
        else:
            by_driver = {}
            by_number = {}
            by_openf1 = {}
            report["capture_reason"] = capture_race.get("reason")
        updated = []
        for row in final.to_pylist():
            jolpica_row = by_driver.get(row["driver_id"])
            number = (
                int(jolpica_row["Driver"]["permanentNumber"])
                if jolpica_row is not None and jolpica_row["Driver"].get("permanentNumber")
                else None
            )
            openf1_row = by_openf1.get(number)
            category, reason = audit_binary_status(row, jolpica_row, openf1_row)
            if category is not None:
                row["dnf_category"] = category.value
                row["dnf"] = category != DnfCategory.FINISHED
                row["audit_reference"] += f":{AUDIT_VERSION}:{capture_hash}"
            updated.append(row)
            report["rows"].append(
                {
                    "driver_id": row["driver_id"],
                    "driver_number": number,
                    "fia_raw_status": row["raw_status"],
                    "jolpica_status": jolpica_row.get("status") if jolpica_row else None,
                    "openf1_flags": {key: openf1_row.get(key) for key in ("dnf", "dns", "dsq")}
                    if openf1_row
                    else None,
                    "binary_dnf": row["dnf"],
                    "reason": reason,
                }
            )
        table = pa.Table.from_pylist(updated, schema=OUTCOME_SCHEMA)
        validate_audited_outcomes(table)
        outcome_path = directory / "outcomes" / f"{event_id.replace('/', '_')}.parquet"
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        if outcome_path.exists():
            if not pq.read_table(outcome_path).equals(table):
                raise ValueError("immutable binary DNF outcome collision")
        else:
            pq.write_table(table, outcome_path, compression="zstd")
        race["original_outcomes"] = original
        race["outcomes"] = {
            "path": outcome_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(outcome_path.read_bytes()).hexdigest(),
        }
        report["source_artifacts"] = sources
        report["binary_labels"] = sum(row["dnf"] is not None for row in updated)
        report["positive_labels"] = sum(row["dnf"] is True for row in updated)
        audit["races"].append(report)
    audit["counts"] = dict(
        Counter(row["reason"] for race in audit["races"] for row in race["rows"])
    )
    audit["binary_labels"] = sum(race["binary_labels"] for race in audit["races"])
    audit["positive_labels"] = sum(race["positive_labels"] for race in audit["races"])
    _immutable(directory / "audit.json", _canonical(audit))
    catalog = directory / "catalog.json"
    _immutable(catalog, _canonical(revised))
    benchmark = build_benchmarks(root, directory / "benchmark", catalog)
    if benchmark["included_races"] != len(revised["races"]):
        raise ValueError("binary DNF benchmark lost an audited Gold race")
    return {
        "benchmark_dir": (directory / "benchmark").relative_to(root).as_posix(),
        "catalog": catalog.relative_to(root).as_posix(),
        "audit": (directory / "audit.json").relative_to(root).as_posix(),
        "races": len(revised["races"]),
        "binary_labels": audit["binary_labels"],
        "positive_labels": audit["positive_labels"],
        "counts": audit["counts"],
    }
