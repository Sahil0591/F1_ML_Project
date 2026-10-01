"""Conservative FIA practice classification audit for historical Gold inputs."""

import hashlib
import json
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from pypdf import PdfReader

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.trust.grid_history import _plain
from f1_ml_predictor.trust.historical import (
    _immutable,
    _json,
    _official,
    _retain_response,
    inspect_pdf,
)
from f1_ml_predictor.trust.winter import DRIVER_ALIASES, _publication, registry_rows


def _session(title: str) -> int | None:
    match = re.fullmatch(
        r"(?:P|Free Practice |(?:First|Second|Third) Practice Session )(\d?) Classification",
        title,
        re.I,
    )
    if match is None:
        return None
    if match[1]:
        return int(match[1])
    return {"first": 1, "second": 2, "third": 3}.get(title.split()[0].lower())


def parse_practice_pdf(
    path: Path, expected_drivers: set[str]
) -> tuple[dict[str, dict[str, Any]], int]:
    """Read ranked FIA timing rows and their printed best laps only."""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    aliases = sorted(
        ((_plain(name), driver) for name, driver in DRIVER_ALIASES.items()),
        key=lambda item: -len(item[0]),
    )
    lines = text.splitlines()
    parsed: dict[str, dict[str, Any]] = {}
    positions: set[int] = set()
    field_positions: set[int] = set()
    for index, raw in enumerate(lines):
        line = _plain(raw)
        match = re.match(r"^(\d{1,2})\s+(\d{1,3})\s+(.+)$", line)
        if match is None:
            continue
        position = int(match[1])
        name = match[3]
        found = {
            driver for alias, driver in aliases if name == alias or name.startswith(alias + " ")
        }
        if len(found) != 1 or not 1 <= position <= 30:
            continue
        driver = found.pop()
        if position in field_positions:
            raise ValueError("practice classification repeats a rank")
        field_positions.add(position)
        if driver not in expected_drivers:
            continue
        if driver in parsed or position in positions:
            raise ValueError("practice classification repeats a Gold roster driver")
        # Ordinary PDF text has a driver line followed by the entrant/timing line.
        # One-line lookahead cannot borrow the next driver's lap.
        tail = lines[index + 1] if index + 1 < len(lines) else ""
        lap = re.search(r"(?<!\d)(\d{1,2}):([0-5]\d)\.(\d{3})(?!\d)", tail)
        seconds = (int(lap[1]) * 60 + int(lap[2]) + int(lap[3]) / 1000) if lap else None
        parsed[driver] = {"position": position, "best_lap_seconds": seconds}
        positions.add(position)
    if len(parsed) < 10 or not field_positions or max(field_positions) < 15:
        raise ValueError("practice PDF lacks a supported ranked timing table")
    return parsed, max(field_positions)


def audit_historical_practice(root: Path, coverage_report: Path) -> dict[str, Any]:
    """Bind latest completed practice document at each existing Gold cutoff."""
    root = root.resolve()
    report = json.loads(coverage_report.read_text(encoding="utf-8"))
    discovery = json.loads(
        _safe_file(root, report["discovery_catalog"], report["discovery_sha256"]).read_text(
            encoding="utf-8"
        )
    )
    direct = json.loads(
        _safe_file(root, report["direct_catalog"], report["direct_catalog_sha256"]).read_text(
            encoding="utf-8"
        )
    )
    by_event = {
        f"season={item['season']}/round={item['round']:02d}": item
        for item in discovery["candidates"]
    }
    direct_by_event = {
        f"season={item['season']}/round={item['round']:02d}": item for item in direct["candidates"]
    }
    registry_artifacts = {
        artifact["url"]: artifact
        for artifact in discovery["source_artifacts"]
        if artifact.get("provider") == "FIA" and artifact.get("path", "").endswith(".html")
    }
    gold = root / "data/benchmarks/gold_core"
    manifest = json.loads((gold / "manifest.json").read_text(encoding="utf-8"))
    if manifest["datasets"]["Gold"]["sha256"] != report["gold_dataset_sha256"]:
        raise ValueError("practice report targets another Gold dataset")
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in pq.read_table(
        _safe_file(gold, "gold.parquet", report["gold_dataset_sha256"])
    ).to_pylist():
        rows_by_event.setdefault(row["event_id"], []).append(row)
    audits = []
    with httpx.Client(
        timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
    ) as client:
        for event_id, rows in sorted(rows_by_event.items()):
            candidate = by_event[event_id]
            cutoff = rows[0]["prediction_timestamp"]
            records = [
                (doc, session)
                for doc in candidate.get("registry_documents", [])
                if (session := _session(doc["title"])) in {1, 2, 3} and doc.get("url")
            ]
            result: dict[str, Any] = {
                "event_id": event_id,
                "cutoff": cutoff.isoformat(),
                "status": "missing",
                "reason": "no_fia_practice_classification",
            }
            available = [
                (doc, session)
                for doc, session in records
                if _publication(doc) + timedelta(minutes=1) <= cutoff
            ]
            if not available:
                if records:
                    result["reason"] = "practice_classification_published_after_cutoff"
                audits.append(result)
                continue
            selected, session = max(
                available,
                key=lambda pair: (pair[1], _publication(pair[0]), int(pair[0]["document_id"] or 0)),
            )
            if selected["recalled"]:
                result["reason"] = "latest_practice_classification_recalled"
                audits.append(result)
                continue
            result.update(
                {"document_id": selected["document_id"], "url": selected["url"], "session": session}
            )
            try:
                _official(selected["url"])
                registry_spec = direct_by_event[event_id].get("research_registry_artifact")
                if registry_spec is None:
                    registry_spec = registry_artifacts[candidate["index_url"]]
                registry_path = _safe_file(root, registry_spec["path"], registry_spec["sha256"])
                matching = [
                    doc
                    for doc in registry_rows(registry_path.read_text(encoding="utf-8"))
                    if doc["url"] == selected["url"]
                    and doc["document_id"] == selected["document_id"]
                    and doc["title"] == selected["title"]
                    and doc["publication_cet"] == selected["publication_cet"]
                    and not doc["recalled"]
                ]
                if len(matching) != 1:
                    raise ValueError("retained FIA registry does not bind the practice version")
                response = client.get(selected["url"])
                artifact = _retain_response(root, response)
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("practice response is not a PDF")
                pdf_path = _safe_file(root, artifact["path"], artifact["sha256"])
                inspected = inspect_pdf(pdf_path)
                cover = _plain(inspected["cover_text"])
                if (
                    inspected["document_id"] is None
                    or selected["document_id"] not in {None, inspected["document_id"]}
                    or str(candidate["season"]) not in cover
                    or "classification" not in cover
                ):
                    raise ValueError("practice PDF cover contradicts the registry version")
                parsed, field_size = parse_practice_pdf(
                    pdf_path, {row["driver_id"] for row in rows}
                )
                result.update(
                    {
                        "status": "audited",
                        "reason": None,
                        "document_id": inspected["document_id"],
                        "available_at": (_publication(selected) + timedelta(minutes=1)).isoformat(),
                        "publication_cet": selected["publication_cet"],
                        "document": {"path": artifact["path"], "sha256": artifact["sha256"]},
                        "registry": {
                            "path": registry_spec["path"],
                            "sha256": registry_spec["sha256"],
                        },
                        "field_size": field_size,
                        "practice": parsed,
                        "audit_method": (
                            "exact FIA registry row, PDF cover identity, "
                            "printed practice rank and lap, roster crosswalk"
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
        / f"practice-audit-{hashlib.sha256(content).hexdigest()}.json"
    )
    _immutable(target, content)
    return {
        "path": target.relative_to(root).as_posix(),
        "sha256": file_sha256(target),
        "audited_races": output["audited_races"],
        "reason_counts": dict(
            sorted(Counter(item["reason"] for item in audits if item["reason"]).items())
        ),
    }
