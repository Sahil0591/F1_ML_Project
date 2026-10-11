"""Audit exact FIA grid publications against frozen historical Gold cutoffs."""

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timedelta
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


def _grid_lines(text: str) -> list[str]:
    """Plain lines, with stacked position / car number / name cells rejoined.

    Some legacy grids print each cell on its own line. Only an exact run of a
    position, a car number and a non-numeric name is joined back into one row.
    """
    lines = [_plain(raw) for raw in text.splitlines()]
    result = []
    index = 0
    while index < len(lines):
        window = lines[index : index + 3]
        if (
            len(window) == 3
            and re.fullmatch(r"\d{1,2}", window[0])
            and re.fullmatch(r"\d{1,3}", window[1])
            and re.match(r"^[a-z]", window[2])
        ):
            result.append(" ".join(window))
            index += 3
            continue
        result.append(lines[index])
        index += 1
    return result


def _pit_lane_lines(text: str) -> list[str]:
    """Lines naming pit-lane starters: inline statements and rows under a pit-lane heading."""
    lines = [_plain(raw) for raw in text.splitlines()]
    result = [line for line in lines if "pit lane" in line]
    for index, line in enumerate(lines):
        if "start from the pit lane" not in line:
            continue
        for row in lines[index + 1 :]:
            if row.startswith("*") or "penalt" in row or "notes" in row:
                break
            if re.match(r"^\d{1,3}\s+[a-z]", row):
                result.append(row)
    return result


def _aliases() -> list[tuple[str, str]]:
    return sorted(
        ((_plain(name), driver) for name, driver in DRIVER_ALIASES.items()),
        key=lambda item: -len(item[0]),
    )


def _grid_rows(text: str, aliases: list[tuple[str, str]]) -> list[tuple[int, str]]:
    """(position, driver) for every explicit grid row naming exactly one known driver."""
    rows = []
    for line in _grid_lines(text):
        match = re.match(r"^(\d{1,2})\s+(\d{1,3})\s+(.+)$", line)
        if match is None:
            continue
        name = match[3]
        found = {
            driver for alias, driver in aliases if name == alias or name.startswith(alias + " ")
        }
        if len(found) == 1:
            rows.append((int(match[1]), found.pop()))
    return rows


def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def grid_drivers(path: Path) -> set[str]:
    """Drivers a grid PDF places on explicit grid rows, before any roster is known."""
    return {driver for _, driver in _grid_rows(_pdf_text(path), _aliases())}


def parse_grid_pdf(path: Path, expected_drivers: set[str]) -> dict[str, dict[str, Any]]:
    """Extract explicit grid ordinals; unsupported layouts remain unaudited."""
    text = _pdf_text(path)
    aliases = _aliases()
    grid: dict[str, dict[str, Any]] = {}
    ordinals: set[int] = set()
    for position, driver in _grid_rows(text, aliases):
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
        pit_lane_lines = _pit_lane_lines(text)
        if not any(
            alias in line
            for line in pit_lane_lines
            for alias, identity in aliases
            if identity == driver
        ):
            raise ValueError("pit-lane statement lacks exact driver identity")
        grid[driver] = {"grid_position": None, "pit_lane_start": True}
    return grid


# Race grid versions only: stewards' "Starting Grid Infringement" decisions and
# sprint grids share the words but are not starting orders for the Grand Prix.
_RACE_GRID_TITLE = re.compile(
    r"^(?:revised )?(?:provisional|final) starting grid(?: \(corrected\))?$"
)
COHORTS = ("post_qualifying", "provisional_grid")


def race_grid_records(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Registry rows that publish the Grand Prix starting order."""
    return [doc for doc in documents if _RACE_GRID_TITLE.match(_plain(doc["title"]))]


def audit_historical_grids(
    root: Path, coverage_report: Path, cohort: str = "post_qualifying"
) -> dict[str, Any]:
    """Fetch only exact registry-listed grid PDFs and freeze audit decisions.

    The post-qualifying cohort keeps the frozen Gold cutoff and accepts a grid only
    if it was published by then. The provisional-grid cohort has its own cutoff per
    race: one minute after the first race grid publication, never earlier than the
    post-qualifying cutoff, so the grid is the one known at that moment.
    """
    if cohort not in COHORTS:
        raise ValueError(f"unknown grid cohort {cohort}")
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
    # Legacy races carry no direct registry artifact; discovery retained their pages.
    registry_artifacts = {
        artifact["url"]: artifact
        for artifact in discovery.get("source_artifacts", [])
        if artifact.get("provider") == "FIA" and artifact.get("path", "").endswith(".html")
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
            post_qualifying = rows[0]["prediction_timestamp"]
            cutoff = post_qualifying
            if cohort == "post_qualifying":
                records = [
                    doc
                    for doc in candidate.get("registry_documents", [])
                    if "starting grid" in doc["title"].lower()
                    and "sprint" not in doc["title"].lower()
                ]
            else:
                records = [
                    doc
                    for doc in race_grid_records(candidate.get("registry_documents", []))
                    if not doc["recalled"]
                ]
                if records:
                    first = min(_publication(doc) for doc in records) + timedelta(minutes=1)
                    cutoff = max(post_qualifying, first)
            result: dict[str, Any] = {
                "event_id": event_id,
                "cutoff": cutoff.isoformat(),
                "status": "missing",
                "reason": "no_fia_race_grid_publication",
            }
            if cohort == "provisional_grid":
                race_start = datetime.fromisoformat(item["race_start"])
                result["post_qualifying_cutoff"] = post_qualifying.isoformat()
                result["race_start"] = race_start.isoformat()
                if cutoff >= race_start:
                    result["cutoff"] = post_qualifying.isoformat()
                    result["reason"] = "grid_published_after_race_start"
                    audits.append(result)
                    continue
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
                    if "final" in _plain(selected["title"]).split()
                    else "provisional",
                }
            )
            try:
                _official(selected["url"])
                registry_spec = item.get("research_registry_artifact")
                if registry_spec is None:
                    registry_spec = registry_artifacts[candidate["index_url"]]
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
                    inspection["document_id"] is None
                    or selected["document_id"] not in {None, inspection["document_id"]}
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
                        # Legacy registry rows omit the number the PDF cover prints.
                        **(
                            {"pdf_document_id": inspection["document_id"]}
                            if selected["document_id"] is None
                            else {}
                        ),
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
        **({"cohort": cohort} if cohort != "post_qualifying" else {}),
        "source_gold_sha256": report["gold_dataset_sha256"],
        "source_coverage_sha256": file_sha256(coverage_report),
        "audited_races": sum(item["status"] == "audited" for item in audits),
        "races": audits,
    }
    content = _json(output)
    target = (
        root
        / "data/benchmarks/historical_audit"
        / f"{'' if cohort == 'post_qualifying' else 'provisional-'}grid-audit-"
        f"{hashlib.sha256(content).hexdigest()}.json"
    )
    _immutable(target, content)
    return {
        "cohort": cohort,
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


def load_provisional_grids(
    root: Path, events: dict[str, tuple[datetime, set[str]]]
) -> tuple[dict[str, dict[str, Any]], str] | None:
    """Verified provisional-grid audit for exactly these Gold races, or None.

    `events` maps each Gold event to its post-qualifying cutoff and roster. The
    newest matching audit is replayed: registry row, publication bound, cutoff
    placement and every parsed ordinal must agree with the retained bytes.
    """
    directory = root / "data/benchmarks/historical_audit"
    matches = []
    for path in directory.glob("provisional-grid-audit-*.json"):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("cohort") == "provisional_grid" and {
            record["event_id"] for record in audit["races"]
        } == set(events):
            matches.append((path.stat().st_mtime, path.name, path, audit))
    if not matches:
        return None
    _, _, path, audit = max(matches)
    digest = file_sha256(path)
    if path.name != f"provisional-grid-audit-{digest}.json":
        raise ValueError("provisional-grid audit name does not match its content hash")
    records: dict[str, dict[str, Any]] = {}
    for record in audit["races"]:
        post_qualifying, roster = events[record["event_id"]]
        cutoff = datetime.fromisoformat(record["cutoff"])
        if record["post_qualifying_cutoff"] != post_qualifying.isoformat():
            raise ValueError("provisional-grid audit targets another post-qualifying cutoff")
        if not post_qualifying <= cutoff < datetime.fromisoformat(record["race_start"]):
            raise ValueError("provisional-grid cutoff is outside the audited race weekend")
        item: dict[str, Any] = {**record, "cutoff": cutoff}
        if record["status"] == "audited":
            document = _safe_file(root, record["document"]["path"], record["document"]["sha256"])
            registry = _safe_file(root, record["registry"]["path"], record["registry"]["sha256"])
            matching = [
                row
                for row in registry_rows(registry.read_text(encoding="utf-8"))
                if row["url"] == record["url"]
                and row["document_id"] == record["document_id"]
                and row["publication_cet"] == record["publication_cet"]
                and not row["recalled"]
                and _RACE_GRID_TITLE.match(_plain(row["title"]))
            ]
            if len(matching) != 1:
                raise ValueError("provisional-grid audit lacks an exact retained FIA registry row")
            available = _publication(matching[0]) + timedelta(minutes=1)
            if available.isoformat() != record["available_at"] or available > cutoff:
                raise ValueError("provisional-grid publication bound differs from the registry")
            if parse_grid_pdf(document, roster) != record["grid"] or set(record["grid"]) != roster:
                raise ValueError("provisional-grid values differ from the exact retained PDF")
            item["available_at"] = available
        records[record["event_id"]] = item
    return records, digest
