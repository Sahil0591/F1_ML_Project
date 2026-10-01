"""Audit exact FIA grid publications against frozen historical Gold cutoffs."""

import hashlib
import json
import re
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from pypdf import PdfReader

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.trust.historical import (
    _immutable,
    _json,
    _official,
    _retain_response,
    inspect_pdf,
)
from f1_ml_predictor.trust.winter import DRIVER_ALIASES, _publication, registry_rows


def _plain(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower().split()
    )


def parse_grid_pdf(path: Path, expected_drivers: set[str]) -> dict[str, dict[str, Any]]:
    """Extract explicit grid ordinals; unsupported layouts remain unaudited."""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    aliases = sorted(
        ((_plain(name), driver) for name, driver in DRIVER_ALIASES.items()),
        key=lambda item: -len(item[0]),
    )
    grid: dict[str, dict[str, Any]] = {}
    ordinals: set[int] = set()
    for raw in text.splitlines():
        line = _plain(raw)
        match = re.match(r"^(\d{1,2})\s+(\d{1,3})\s+(.+)$", line)
        if match is None:
            continue
        position = int(match[1])
        name = match[3]
        found = {
            driver for alias, driver in aliases if name == alias or name.startswith(alias + " ")
        }
        if len(found) != 1:
            continue
        driver = found.pop()
        if driver not in expected_drivers or driver in grid or position in ordinals:
            raise ValueError("grid PDF has duplicate or non-roster driver/ordinal")
        grid[driver] = {"grid_position": position, "pit_lane_start": False}
        ordinals.add(position)
    if len(grid) < max(15, len(expected_drivers) - 2) or ordinals != set(
        range(1, len(ordinals) + 1)
    ):
        raise ValueError("grid PDF does not yield a complete contiguous starting order")
    if len(grid) != len(expected_drivers):
        # An omitted entrant can be a pit-lane start, withdrawal, or a parser miss.
        # The document must say so explicitly before assigning that status.
        missing = expected_drivers - set(grid)
        if len(missing) != 1:
            raise ValueError("grid PDF omits roster drivers without an explicit pit-lane statement")
        driver = missing.pop()
        pit_lane_lines = [_plain(line) for line in text.splitlines() if "pit lane" in _plain(line)]
        if not any(
            alias in line
            for line in pit_lane_lines
            for alias, identity in aliases
            if identity == driver
        ):
            raise ValueError("pit-lane statement lacks exact driver identity")
        grid[driver] = {"grid_position": None, "pit_lane_start": True}
    return grid


def audit_historical_grids(root: Path, coverage_report: Path) -> dict[str, Any]:
    """Fetch only exact registry-listed grid PDFs and freeze audit decisions."""
    root = root.resolve()
    report = json.loads(coverage_report.read_text(encoding="utf-8"))
    discovery_path = _safe_file(root, report["discovery_catalog"], report["discovery_sha256"])
    direct_path = _safe_file(root, report["direct_catalog"], report["direct_catalog_sha256"])
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    by_event = {
        f"season={item['season']}/round={item['round']:02d}": item
        for item in discovery["candidates"]
    }
    direct_by_event = {
        f"season={item['season']}/round={item['round']:02d}": item for item in direct["candidates"]
    }
    gold = root / "data/benchmarks/gold_core"
    gold_manifest = json.loads((gold / "manifest.json").read_text(encoding="utf-8"))
    if report["gold_dataset_sha256"] != gold_manifest["datasets"]["Gold"]["sha256"]:
        raise ValueError("grid report does not match the frozen Gold dataset")
    source = _safe_file(gold, "gold.parquet", report["gold_dataset_sha256"])
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in pq.read_table(source).to_pylist():
        rows_by_event.setdefault(row["event_id"], []).append(row)
    audits = []
    with httpx.Client(
        timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
    ) as client:
        for event_id, rows in sorted(rows_by_event.items()):
            candidate = by_event[event_id]
            item = direct_by_event[event_id]
            cutoff = rows[0]["prediction_timestamp"]
            records = [
                doc
                for doc in candidate.get("registry_documents", [])
                if "starting grid" in doc["title"].lower() and "sprint" not in doc["title"].lower()
            ]
            result: dict[str, Any] = {
                "event_id": event_id,
                "cutoff": cutoff.isoformat(),
                "status": "missing",
                "reason": "no_fia_race_grid_publication",
            }
            if not records:
                audits.append(result)
                continue
            available = [
                doc for doc in records if _publication(doc) + timedelta(minutes=1) <= cutoff
            ]
            if not available:
                result["reason"] = "grid_published_after_post_qualifying_cutoff"
                result["first_grid_publication"] = min(
                    _publication(doc) + timedelta(minutes=1) for doc in records
                ).isoformat()
                audits.append(result)
                continue
            selected = max(
                available, key=lambda doc: (_publication(doc), int(doc["document_id"] or 0))
            )
            if selected["recalled"]:
                result["reason"] = "latest_grid_publication_recalled"
                audits.append(result)
                continue
            result.update(
                {
                    "document_id": selected["document_id"],
                    "url": selected["url"],
                    "grid_status": "final"
                    if selected["title"].lower().startswith("final")
                    else "provisional",
                }
            )
            try:
                _official(selected["url"])
                registry_spec = item["research_registry_artifact"]
                registry_path = _safe_file(root, registry_spec["path"], registry_spec["sha256"])
                matching = [
                    doc
                    for doc in registry_rows(registry_path.read_text(encoding="utf-8"))
                    if doc["url"] == selected["url"]
                    and doc["document_id"] == selected["document_id"]
                    and doc["publication_cet"] == selected["publication_cet"]
                    and doc["title"] == selected["title"]
                    and not doc["recalled"]
                ]
                if len(matching) != 1:
                    raise ValueError("retained FIA registry does not bind the exact grid version")
                response = client.get(selected["url"])
                artifact = _retain_response(root, response)
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("grid response is not a PDF")
                pdf_path = _safe_file(root, artifact["path"], artifact["sha256"])
                inspection = inspect_pdf(pdf_path)
                cover = _plain(inspection["cover_text"])
                if (
                    inspection["document_id"] != selected["document_id"]
                    or str(candidate["season"]) not in cover
                    or "starting grid" not in cover
                ):
                    raise ValueError("grid PDF cover contradicts the registry version")
                parsed = parse_grid_pdf(pdf_path, {row["driver_id"] for row in rows})
                result.update(
                    {
                        "status": "audited",
                        "reason": None,
                        "available_at": (_publication(selected) + timedelta(minutes=1)).isoformat(),
                        "publication_cet": selected["publication_cet"],
                        "document": {"path": artifact["path"], "sha256": artifact["sha256"]},
                        "registry": {
                            "path": registry_spec["path"],
                            "sha256": registry_spec["sha256"],
                        },
                        "grid": parsed,
                        "audit_method": (
                            "exact FIA registry row, PDF cover identity, "
                            "complete explicit ordinal parse, roster crosswalk"
                        ),
                    }
                )
            except (OSError, KeyError, ValueError, httpx.HTTPError) as exc:
                result["reason"] = str(exc)
            audits.append(result)
    output = {
        "version": 1,
        "source_gold_sha256": report["gold_dataset_sha256"],
        "source_coverage_sha256": file_sha256(coverage_report),
        "audited_races": sum(item["status"] == "audited" for item in audits),
        "races": audits,
    }
    content = _json(output)
    target = (
        root
        / "data/benchmarks/historical_audit"
        / f"grid-audit-{hashlib.sha256(content).hexdigest()}.json"
    )
    _immutable(target, content)
    return {
        "path": target.relative_to(root).as_posix(),
        "sha256": file_sha256(target),
        "audited_races": output["audited_races"],
        "reason_counts": dict(
            sorted(
                (reason, sum(item["reason"] == reason for item in audits))
                for reason in {item["reason"] for item in audits if item["reason"]}
            )
        ),
    }
