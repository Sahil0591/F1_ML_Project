"""Direct FIA audit of historical sprint grids and sprint classifications (Gold).

For every completed sprint weekend in a retained scoring-audit collection, the audit
reads the event's FIA document registry and:

- selects the sprint grid document: the first non-recalled sprint qualifying
  classification (2024 on), sprint shootout classification (2023) or, in 2022, the
  main qualifying classification, which set the sprint grid that season. The cutoff
  is its registry publication clock read as the later UTC bound plus one minute,
  as in the race Gold audit;
- requires that cutoff to precede the Final Sprint Starting Grid publication, which
  the FIA issues before the start, so the grid is known before the sprint;
- takes labels from the latest non-recalled Final Sprint Classification and refuses
  an event with a later sprint ruling that could amend it;
- cross-checks the classification against the FIA sprint positions already audited
  in the scoring ledger, and labels retirements only on three-source agreement
  (FIA, Jolpica, OpenF1) with the unchanged binary DNF rule.

Retained PDFs are content addressed. A sprint failing any step is excluded with its
reason; nothing is inferred to fill a gap.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.binary_dnf import audit_binary_status
from f1_ml_predictor.trust.fia_tables import parse_final_text, parse_qualifying_text
from f1_ml_predictor.trust.historical import _official, _retain_response, inspect_pdf
from f1_ml_predictor.trust.winter import (
    DRIVER_ALIASES,
    _publication,
    _version_key,
    constructor_aliases_for_season,
    registry_rows,
)

SPRINT_GOLD_VERSION = "gold-sprint-core-v1"
AUDIT_METHOD = "fia-sprint-direct-v2-cet-upper-bound"
_ROOT = Path("data/benchmarks/gold_sprint_core_v1")
_FINAL = re.compile(r"^final sprint classification$", re.I)
_STARTING_GRID = re.compile(r"^(?:provisional |final )?sprint (?:qualifying )?starting grid$", re.I)
_GRID = {
    "sprint_qualifying": re.compile(
        r"^(?:provisional |final )?sprint qualifying classification$", re.I
    ),
    "sprint_shootout": re.compile(
        r"^(?:provisional |final )?sprint shootout classification$", re.I
    ),
    "qualifying": re.compile(r"^(?:provisional |final )?qualifying classification$", re.I),
}
# Documents after the final sprint classification that could amend it. Late uploads of
# grid documents (sprint qualifying classification, starting grid) cannot.
_RULING = re.compile(r"decision|infringement|penalt|review|summons|offence|classification", re.I)
_GRID_DOCUMENT = re.compile(
    r"qualifying classification|shootout classification|starting grid", re.I
)
# Later sprint rulings read against the Final Sprint Classification and found not to
# amend it, bound to the ruling PDF. A changed PDF excludes the sprint again.
REVIEWED_LATER_RULINGS = {
    "https://www.fia.com/system/files/decision-document/2025_sao_paulo_grand_prix_-_corrected_-"
    "_sprint_infringement_-_car_30_-_causing_a_collision_with_car_87_at_t4.pdf": {
        "sha256": "e8c6a44309ce67ee3eb755e37fc8e71c4068ab7c22544b617b58ce29c7abe3d3",
        "finding": "corrected reissue of recalled document 41 with the same 5 second time "
        "penalty for car 30; the Final Sprint Classification (document 42) already applies "
        "it and cites document 41",
    },
}
SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string(), nullable=False),
        pa.field("sprint_qualifying_position", pa.int16()),
        pa.field("sprint_qualifying_last_seconds", pa.float64()),
        pa.field("grid_available_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("label_position", pa.int16()),
        pa.field("label_classified", pa.bool_(), nullable=False),
        pa.field("label_dnf", pa.bool_()),
        pa.field("label_dnf_reason", pa.string(), nullable=False),
        pa.field("label_available_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def grid_kind(season: int) -> str:
    if season <= 2022:
        return "qualifying"
    return "sprint_shootout" if season == 2023 else "sprint_qualifying"


def _available(row: dict[str, Any]) -> datetime:
    return _publication(row) + timedelta(minutes=1)


def _later_rulings(live: list[dict[str, Any]], final: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in live
        if _version_key(row) > _version_key(final)
        and "sprint" in row["title"].lower()
        and _RULING.search(row["title"])
        and not _GRID_DOCUMENT.search(row["title"])
        and "championship points" not in row["title"].lower()
    ]


def select_documents(
    rows: list[dict[str, Any]], season: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Grid, starting-grid and final classification rows; raises with an exclusion reason."""
    live = [row for row in rows if row.get("url") and not row["recalled"]]
    grids = sorted(
        (row for row in live if _GRID[grid_kind(season)].match(row["title"])), key=_version_key
    )
    if not grids:
        raise ValueError("fia_sprint_grid_document_missing")
    grid = grids[0]
    starting = sorted((row for row in live if _STARTING_GRID.match(row["title"])), key=_version_key)
    if not starting:
        raise ValueError("fia_final_sprint_starting_grid_missing")
    if _available(grid) > _publication(starting[0]):
        raise ValueError("sprint_grid_not_published_before_the_starting_grid")
    finals = sorted((row for row in live if _FINAL.match(row["title"])), key=_version_key)
    if not finals:
        raise ValueError("fia_final_sprint_classification_missing")
    final = finals[-1]
    later = [row for row in _later_rulings(live, final) if row["url"] not in REVIEWED_LATER_RULINGS]
    if later:
        raise ValueError(
            "later_sprint_ruling_requires_review:" + "|".join(r["title"] for r in later)
        )
    return grid, starting[0], final


def _document(
    root: Path, client: httpx.Client, row: dict[str, Any], event: EventId
) -> tuple[dict[str, Any], dict[str, Any]]:
    _official(row["url"])
    artifact = _retain_response(root, client.get(row["url"]))
    pdf = inspect_pdf(root / artifact["path"])
    if row["document_id"] is not None and pdf["document_id"] not in {None, row["document_id"]}:
        raise ValueError("pdf_document_number_contradicts_registry")
    if str(event.season) not in pdf["cover_text"] + pdf["text"][:4000]:
        raise ValueError("pdf_cover_does_not_name_the_season")
    return artifact, pdf


def _binding(row: dict[str, Any], artifact: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "role": role,
        "title": row["title"],
        "document_id": row["document_id"],
        "url": row["url"],
        "publication_cet": row["publication_cet"],
        "available_at_utc": _available(row).isoformat(),
        "path": artifact["path"],
        "sha256": artifact["sha256"],
    }


def audit_event(
    root: Path,
    client: httpx.Client,
    event: EventId,
    circuit_id: str,
    registry: dict[str, Any],
    points: dict[str, Any],
    jolpica: dict[str, dict[str, Any]],
    openf1: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows and provenance for one sprint; raises ValueError with an exclusion reason."""
    rows = registry_rows((root / registry["path"]).read_text(encoding="utf-8"))
    grid_row, starting_row, final_row = select_documents(rows, event.season)
    constructors = constructor_aliases_for_season(event.season)
    grid_artifact, grid_pdf = _document(root, client, grid_row, event)
    final_artifact, final_pdf = _document(root, client, final_row, event)
    reviewed = []
    live = [row for row in rows if row.get("url") and not row["recalled"]]
    for ruling in _later_rulings(live, final_row):
        artifact, _ = _document(root, client, ruling, event)
        if artifact["sha256"] != REVIEWED_LATER_RULINGS[ruling["url"]]["sha256"]:
            raise ValueError(f"reviewed_later_sprint_ruling_changed:{ruling['title']}")
        reviewed.append(
            {
                **_binding(ruling, artifact, "reviewed_later_ruling"),
                "finding": REVIEWED_LATER_RULINGS[ruling["url"]]["finding"],
            }
        )
    grid = {
        row["driver_id"]: row
        for row in parse_qualifying_text(
            grid_pdf["text"], event, DRIVER_ALIASES, constructors
        ).to_pylist()
    }
    final = {
        row["driver_id"]: row
        for row in parse_final_text(
            final_pdf["text"], event, DRIVER_ALIASES, constructors
        ).to_pylist()
    }
    if set(grid) - set(final):
        # A driver on the grid document with no classification row has no label.
        raise ValueError("sprint_grid_driver_missing_from_classification")
    for driver, row in final.items():
        # A classified driver who set no sprint qualifying time is absent from the grid
        # document; the classification proves the entry, and the grid stays missing.
        grid.setdefault(
            driver,
            {
                "constructor_id": row["constructor_id"],
                "position": None,
                "q1_seconds": None,
                "q2_seconds": None,
                "q3_seconds": None,
            },
        )
        if row["constructor_id"] != grid[driver]["constructor_id"]:
            raise ValueError(f"constructor_changed_between_grid_and_sprint:{driver}")
        audited = points.get(driver)
        if audited is None:
            raise ValueError(f"no_fia_sprint_points_record:{driver}")
        position = audited.get("sprint_position")
        if position is not None and str(position).isdecimal():
            if row["position"] != int(position):
                raise ValueError(f"fia_classification_disagrees_with_points_audit:{driver}")
    grid_at = _available(grid_row)
    label_at = _available(final_row)
    output = []
    reasons: dict[str, str] = {}
    for driver in sorted(final):
        row = final[driver]
        times = (grid[driver]["q3_seconds"], grid[driver]["q2_seconds"], grid[driver]["q1_seconds"])
        category, reason = audit_binary_status(
            row,
            jolpica.get(driver),
            None if openf1 is None else openf1.get(driver),
        )
        label_dnf = None if category is None else category.value != "finished"
        reasons[driver] = reason
        output.append(
            {
                "event_id": event.partition(),
                "driver_id": driver,
                "constructor_id": row["constructor_id"],
                "sprint_qualifying_position": grid[driver]["position"],
                "sprint_qualifying_last_seconds": next((t for t in times if t), None),
                "grid_available_at": grid_at,
                "label_position": row["position"],
                "label_classified": row["classified"],
                "label_dnf": label_dnf,
                "label_dnf_reason": reason,
                "label_available_at": label_at,
            }
        )
    provenance = {
        "event_id": event.partition(),
        "circuit_id": circuit_id,
        "grid_kind": grid_kind(event.season),
        "grid_available_at_utc": grid_at.isoformat(),
        "label_available_at_utc": label_at.isoformat(),
        "starting_grid_publication_cet": starting_row["publication_cet"],
        "registry": {"path": registry["path"], "sha256": registry["sha256"]},
        "documents": [
            _binding(grid_row, grid_artifact, "sprint_grid"),
            _binding(final_row, final_artifact, "sprint_final_classification"),
            *reviewed,
        ],
        "dnf_labels": sum(item["label_dnf"] is not None for item in output),
        "audit_method": AUDIT_METHOD,
    }
    return output, provenance


def build_gold_sprints(
    root: Path,
    collection_path: Path,
    *,
    sources: Callable[[EventId], tuple[dict[str, Any], dict[str, Any] | None]],
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Audit every completed sprint in a scoring collection and freeze a Gold version.

    ``sources`` returns the Jolpica and OpenF1 sprint status rows for an event, keyed
    by canonical driver, used only for the three-source retirement agreement.
    """
    root = root.resolve()
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    evidence = json.loads(
        (root / "data/audit/event_points_evidence.json").read_text(encoding="utf-8")
    )
    points: dict[tuple[int, int], dict[str, Any]] = {}
    for record in evidence["records"]:
        points.setdefault((record["season"], record["round"]), {})[record["driver"]] = record
    client = http_client or httpx.Client(
        timeout=60, follow_redirects=True, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
    )
    rows: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    try:
        for season in sorted(collection["seasons"], key=int):
            for item in collection["seasons"][season]["events"]:
                if not item.get("has_sprint") or not item.get("completed"):
                    continue
                event = EventId(item["season"], item["round"])
                try:
                    if item.get("fia_event_page") is None:
                        raise ValueError("fia_event_registry_missing")
                    jolpica, openf1 = sources(event)
                    built, provenance = audit_event(
                        root,
                        client,
                        event,
                        item.get("circuit_id", ""),
                        item["fia_event_page"],
                        points.get((event.season, event.round), {}),
                        jolpica,
                        openf1,
                    )
                except (ValueError, KeyError, RuntimeError, OSError, httpx.HTTPError) as exc:
                    excluded.append({"event_id": event.partition(), "reason": str(exc)})
                    continue
                rows.extend(built)
                included.append(provenance)
    finally:
        if http_client is None:
            client.close()
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    staging = root / _ROOT / ".staging.parquet"
    staging.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, staging)
    data_sha = file_sha256(staging)
    manifest = {
        "version": 1,
        "dataset_version": SPRINT_GOLD_VERSION,
        "tier": "Gold",
        "audit_method": AUDIT_METHOD,
        "collection": {"path": collection_path.as_posix(), "sha256": file_sha256(collection_path)},
        "event_points_evidence_sha256": file_sha256(root / "data/audit/event_points_evidence.json"),
        "dataset": {"path": "sprints.parquet", "sha256": data_sha, "rows": table.num_rows},
        "included": included,
        "excluded": excluded,
    }
    data = _canonical(manifest)
    digest = hashlib.sha256(data).hexdigest()
    directory = root / _ROOT / digest
    if directory.exists():
        staging.unlink()
        if file_sha256(directory / "sprints.parquet") != data_sha:
            raise ValueError("immutable Gold sprint dataset collision")
    else:
        directory.mkdir(parents=True)
        staging.replace(directory / "sprints.parquet")
        with (directory / "manifest.json").open("xb") as handle:
            handle.write(data)
    pointer = root / _ROOT / "current.json"
    pointer.write_text(
        json.dumps(
            {"manifest_sha256": digest, "recorded_at": datetime.now(UTC).isoformat()}, indent=2
        ),
        encoding="utf-8",
    )
    return {
        "path": directory.relative_to(root).as_posix(),
        "manifest_sha256": digest,
        "included": len(included),
        "excluded": excluded,
        "rows": table.num_rows,
        "dnf_labels": sum(row["label_dnf"] is not None for row in rows),
    }


def load_gold_sprints(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str] | None:
    """The current hash-verified Gold sprint version, or None when none was built."""
    pointer = root / _ROOT / "current.json"
    if not pointer.exists():
        return None
    digest = json.loads(pointer.read_text(encoding="utf-8"))["manifest_sha256"]
    directory = root / _ROOT / digest
    data = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Gold sprint manifest hash mismatch")
    manifest = json.loads(data)
    if file_sha256(directory / "sprints.parquet") != manifest["dataset"]["sha256"]:
        raise ValueError("Gold sprint dataset hash mismatch")
    return pq.read_table(directory / "sprints.parquet").to_pylist(), manifest, digest
