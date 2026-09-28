"""Direct FIA winter-publication audits over a bounded researched candidate pool."""

import hashlib
import json
import re
import time
from datetime import UTC, datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.features.contracts import PreRaceEvent
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.evidence import table_hash
from f1_ml_predictor.trust.f1_schedule import validate_f1_schedule
from f1_ml_predictor.trust.fia import (
    FiaDocumentMetadata,
    FiaDocumentStatus,
    fia_publication_evidence,
)
from f1_ml_predictor.trust.fia_tables import (
    ROSTER_SCHEMA,
    parse_final_text,
    parse_qualifying_text,
    parse_roster_text,
)
from f1_ml_predictor.trust.historical import (
    _immutable,
    _json,
    _official,
    _retain_response,
    inspect_pdf,
    reconstruct_gold_core,
)
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION, OUTCOME_SCHEMA
from f1_ml_predictor.trust.transcription import reviewed_qualifying

# These aliases identify entities, not entrant membership. Membership is read
# from each contemporary PDF's entrant column. Unmapped names fail explicitly.
_DRIVERS = {
    "Max VERSTAPPEN": "max_verstappen",
    "Lando NORRIS": "norris",
    "Oscar PIASTRI": "piastri",
    "George RUSSELL": "russell",
    "Charles LECLERC": "leclerc",
    "Lewis HAMILTON": "hamilton",
    "Fernando ALONSO": "alonso",
    "Lance STROLL": "stroll",
    "Carlos SAINZ": "sainz",
    "Alexander ALBON": "albon",
    "Alex ALBON": "albon",
    "Pierre GASLY": "gasly",
    "Esteban OCON": "ocon",
    "Oliver BEARMAN": "bearman",
    "Nico HULKENBERG": "hulkenberg",
    "Nico HÜLKENBERG": "hulkenberg",
    "Gabriel BORTOLETO": "bortoleto",
    "Isack HADJAR": "hadjar",
    "Yuki TSUNODA": "tsunoda",
    "Liam LAWSON": "lawson",
    "Jack DOOHAN": "doohan",
    "Franco COLAPINTO": "colapinto",
    "Kimi ANTONELLI": "antonelli",
    "Andrea Kimi ANTONELLI": "antonelli",
    "Valtteri BOTTAS": "bottas",
    "Sergio PEREZ": "perez",
    "Sergio PÉREZ": "perez",
    "Arvid LINDBLAD": "lindblad",
}
_CONSTRUCTORS = {
    "Oracle Red Bull Racing": "red_bull",
    "Red Bull Racing": "red_bull",
    "McLaren Formula 1 Team": "mclaren",
    "McLaren Mastercard F1 Team": "mclaren",
    "McLaren": "mclaren",
    "Mercedes-AMG PETRONAS F1 Team": "mercedes",
    "Mercedes": "mercedes",
    "Scuderia Ferrari HP": "ferrari",
    "Ferrari": "ferrari",
    "Aston Martin Aramco F1 Team": "aston_martin",
    "Aston Martin Aramco Mercedes": "aston_martin",
    "Atlassian Williams Racing": "williams",
    "Atlassian Williams F1 Team": "williams",
    "Williams Mercedes": "williams",
    "Atlassian Williams Mercedes": "williams",
    "BWT Alpine F1 Team": "alpine",
    "Alpine Renault": "alpine",
    "Alpine Mercedes": "alpine",
    "MoneyGram Haas F1 Team": "haas",
    "TGR Haas F1 Team": "haas",
    "Haas Ferrari": "haas",
    "Visa Cash App Racing Bulls F1 Team": "rb",
    "Visa Cash App R acing Bulls F1 Team": "rb",
    "Racing Bulls Honda RBPT": "rb",
    "Visa Cash App RB Formula One Team": "rb",
    "Racing Bulls Red Bull Ford": "rb",
    "Kick Sauber F1 Team": "sauber",
    "Stake F1 Team Kick Sauber": "sauber",
    "Kick Sauber Ferrari": "sauber",
    "Sauber": "sauber",
    "Audi Revolut F1 Team": "audi",
    "Audi": "audi",
    "Cadillac Formula 1 Team": "cadillac",
    "Cadillac Ferrari": "cadillac",
    "Red Bull Racing Honda RBPT": "red_bull",
    "Red Bull Racing Red Bull Ford": "red_bull",
    "McLaren Mercedes": "mclaren",
    "Aston Martin Aramco Honda": "aston_martin",
}
DRIVER_ALIASES = {**_DRIVERS, **{key.title(): value for key, value in _DRIVERS.items()}}
CONSTRUCTOR_ALIASES = _CONSTRUCTORS


def registry_rows(text: str) -> list[dict[str, Any]]:
    """Keep recalled rows as well as downloadable document publication records."""
    rows = []
    for block in re.findall(r'<li\b[^>]*class="document-row[^>]*>.*?</li>', text, re.S):
        plain = " ".join(unescape(re.sub(r"<[^>]+>", " ", block)).split())
        match = re.search(
            r"Doc\s+(\d+)\s*-\s*(.*?)\s+Published on\s+(\d{2}\.\d{2}\.\d{2}\s+\d{2}:\d{2})\s+CET",
            plain,
        )
        if not match:
            continue
        url = re.search(r'<a\b[^>]*href="([^"]+)"', block)
        rows.append(
            {
                "document_id": match[1],
                "title": match[2],
                "publication_cet": match[3],
                "url": "https://www.fia.com" + unescape(url[1])
                if url and url[1].startswith("/")
                else (unescape(url[1]) if url else None),
                "recalled": "recalled" in plain.lower(),
            }
        )
    if not rows:
        raise ValueError("FIA registry contains no supported exact publication records")
    return rows


def _publication(row: dict[str, Any], *, winter_required: bool = True) -> datetime:
    local = datetime.strptime(row["publication_cet"], "%d.%m.%y %H:%M")
    if winter_required and local.replace(tzinfo=ZoneInfo("Europe/Paris")).utcoffset() != timedelta(
        hours=1
    ):
        raise ValueError("summer publication clock remains unresolved for predictive evidence")
    return local.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC)


def _record(rows: list[dict[str, Any]], url: str, identifier: str) -> dict[str, Any]:
    _official(url)
    matching = [row for row in rows if row["url"] == url and row["document_id"] == identifier]
    if len(matching) != 1 or matching[0]["recalled"]:
        raise ValueError("exact downloadable document version is missing, recalled or ambiguous")
    return matching[0]


def qualifying_at_cutoff(
    rows: list[dict[str, Any]], selected: dict[str, Any], cutoff: datetime
) -> None:
    """Reject another main qualifying version; sprint classification is separate."""
    known = [
        row
        for row in rows
        if "qualifying classification" in row["title"].lower()
        and "sprint" not in row["title"].lower()
        and _publication(row) <= cutoff
    ]
    if (
        not known
        or selected["recalled"]
        or _publication(selected) + timedelta(minutes=1) > cutoff
        or max(known, key=lambda row: (_publication(row), int(row["document_id"]))) != selected
    ):
        raise ValueError("required qualifying state at cutoff includes another version")


def roster_at_cutoff(
    rows: list[dict[str, Any]], selected: dict[str, Any], cutoff: datetime, *, from_qualifying: bool
) -> None:
    """A possibly published newer entry list blocks an older roster derivation."""
    known = [
        row for row in rows if "entry list" in row["title"].lower() and _publication(row) <= cutoff
    ]
    if not known:
        if not from_qualifying:
            raise ValueError("entry-list roster is not present at cutoff")
        return
    latest = max(known, key=lambda row: (_publication(row), int(row["document_id"])))
    if from_qualifying:
        if _publication(latest) >= _publication(selected):
            raise ValueError("newer entry-list state requires roster review")
    elif (
        latest != selected
        or selected["recalled"]
        or _publication(selected) + timedelta(minutes=1) > cutoff
    ):
        raise ValueError("selected roster is not the latest known entry-list state")


def latest_final_record(rows: list[dict[str, Any]], url: str, identifier: str) -> dict[str, Any]:
    """Require the latest final classification to cover the last event ruling."""
    selected = _record(rows, url, identifier)
    versions = [
        row
        for row in rows
        if "final race classification" in row["title"].lower()
        and "sprint" not in row["title"].lower()
        and not row["recalled"]
    ]
    if (
        not versions
        or max(
            versions,
            key=lambda row: (_publication(row, winter_required=False), int(row["document_id"])),
        )
        != selected
    ):
        raise ValueError("selected final target is not the latest official classification")
    event_prefix = url.split("_-_", 1)[0] + "_-_"
    later = [
        row
        for row in rows
        if not row["recalled"]
        and row.get("url")
        and row["url"].startswith(event_prefix)
        and (_publication(row, winter_required=False), int(row["document_id"]))
        > (_publication(selected, winter_required=False), int(selected["document_id"]))
        and row["title"].lower() != "championship points"
    ]
    if later:
        raise ValueError("later event documents require final outcome review")
    return selected


def _proof(
    item: dict[str, Any],
    row: dict[str, Any],
    artifact: dict[str, Any],
    table: pa.Table,
    status: FiaDocumentStatus,
    audit: str,
) -> Any:
    published = _publication(row)
    return fia_publication_evidence(
        FiaDocumentMetadata(
            item["url"],
            str(item["document_id"]),
            artifact["sha256"],
            datetime.strptime(row["publication_cet"], "%d.%m.%y %H:%M").strftime("%Y-%m-%d %H:%M"),
            "CET",
            published,
            status,
            table_hash(table),
            True,
            audit,
            60,
        ),
        table,
    )


def _resume_request(root: Path, item: dict[str, Any]) -> Path | None:
    """Recheck a frozen audit without downloading replacement source bytes."""
    index = root / "data/benchmarks/gold_core_registry.json"
    if not index.exists():
        return None
    event = EventId(item["season"], item["round"]).partition()
    cutoff = datetime.fromisoformat(item["prediction_timestamp_utc"])
    matches = [
        row
        for row in json.loads(index.read_text(encoding="utf-8"))["races"]
        if row["event_id"] == event
        and datetime.fromisoformat(row["prediction_timestamp"]) == cutoff
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("ambiguous frozen Core cohort")
    race = matches[0]
    directory = root / "data/features/historical_evidence" / str(race["features"]["sha256"])
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if any(
        not binding["audit_reference"].startswith("fia-winter-direct-v2:")
        for binding in request["document_bindings"]
    ):
        raise ValueError("older winter audit must be withdrawn and rechecked")
    if request["event"]["circuit_id"] != item["circuit_id"]:
        raise ValueError("frozen cohort contradicts candidate circuit")
    if race.get("outcomes") is None:
        return None
    target = json.loads(
        (directory / "targets" / f"{race['outcomes']['sha256']}.json").read_text(encoding="utf-8")
    )
    request["outcomes"] = target["outcomes"]
    request["outcome_document_bindings"] = target["document_bindings"]
    path = directory / f"replay-{hashlib.sha256(_json(request)).hexdigest()}.json"
    _immutable(path, _json(request))
    return path


def audit_winter_pool(
    catalog_path: Path,
    root: Path,
    *,
    minimum_races: int = 8,
    http_client: httpx.Client | None = None,
    reuse_retained: bool = False,
) -> dict[str, Any]:
    """Audit exact document values and stop after enough complete Gold joins.

    No current API classifications are used. Independent source downloads remain
    current-state artifacts; only the checked exact document/table association is
    assigned direct audited publication evidence. Optional features stay missing.
    """
    root = root.resolve()
    candidates = json.loads(catalog_path.read_text(encoding="utf-8"))["candidates"]
    if not 8 <= minimum_races <= 20 or not 1 <= len(candidates) <= 20:
        raise ValueError("use a bounded pool and at least eight eligible races")
    index = root / "data/benchmarks/gold_core_registry.json"
    registered = (
        {row["event_id"] for row in json.loads(index.read_text(encoding="utf-8"))["races"]}
        if index.exists()
        else set()
    )
    candidates.sort(
        key=lambda item: (
            EventId(item["season"], item["round"]).partition() not in registered,
            -item["season"],
            -item["round"],
        )
    )
    client = http_client or httpx.Client(
        timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
    )
    own = http_client is None
    results: list[dict[str, Any]] = []
    retained: dict[str, dict[str, Any]] = {}
    if reuse_retained:
        for path in (root / "data/raw/fia_audit/requests").glob("*.json"):
            artifact = json.loads(path.read_text(encoding="utf-8"))
            captured = datetime.fromisoformat(artifact["captured_at_utc"])
            if timedelta(0) <= datetime.now(UTC) - captured <= timedelta(days=1):
                previous = retained.get(artifact["url"])
                if previous is None or captured > datetime.fromisoformat(
                    previous["captured_at_utc"]
                ):
                    _safe_file(root, artifact["path"], artifact["sha256"])
                    retained[artifact["url"]] = artifact

    def fetch(url: str) -> dict[str, Any]:
        if url not in retained:
            if own:
                time.sleep(1)
            retained[url] = _retain_response(root, client.get(url))
        return retained[url]

    try:
        for item in candidates:
            event = EventId(item["season"], item["round"])
            result: dict[str, Any] = {"event_id": event.partition(), "status": "excluded"}
            try:
                resumed = _resume_request(root, item)
                if resumed is not None:
                    outcome = reconstruct_gold_core(resumed, root)
                    result.update(
                        {
                            "status": "included",
                            "snapshot_id": outcome["snapshot_id"],
                            "request": resumed.relative_to(root).as_posix(),
                            "evidence_manifest": outcome["evidence_manifest"],
                            "reasons": [],
                            "replayed_frozen_audit": True,
                        }
                    )
                    results.append(result)
                    if sum(row["status"] == "included" for row in results) >= minimum_races:
                        break
                    continue
                _official(item["index_url"])
                registry = fetch(item["index_url"])
                rows = registry_rows((root / registry["path"]).read_text(encoding="utf-8"))
                q_spec = {"url": item["qualifying_url"], "document_id": str(item["document_id"])}
                q_row = _record(rows, q_spec["url"], q_spec["document_id"])
                published = _publication(q_row)
                if published != datetime.fromisoformat(item["published_at_utc"]):
                    raise ValueError("research publication claim differs from retained registry")
                cutoff = datetime.fromisoformat(item["prediction_timestamp_utc"])
                qualifying_at_cutoff(rows, q_row, cutoff)
                q_artifact = fetch(q_spec["url"])
                q_text = inspect_pdf(root / q_artifact["path"])
                if q_text["document_id"] != q_spec["document_id"]:
                    raise ValueError(
                        "downloaded qualifying PDF differs from registered document number"
                    )
                review = item.get("qualifying_transcription")
                qualifying = (
                    reviewed_qualifying(
                        q_text["text"],
                        event,
                        DRIVER_ALIASES,
                        CONSTRUCTOR_ALIASES,
                        review,
                        q_artifact["sha256"],
                    )
                    if review
                    else parse_qualifying_text(
                        q_text["text"], event, DRIVER_ALIASES, CONSTRUCTOR_ALIASES
                    )
                )
                audit = f"fia-winter-direct-v1:{registry['sha256']}:{q_artifact['sha256']}"
                q_proof = _proof(
                    q_spec, q_row, q_artifact, qualifying, FiaDocumentStatus.PROVISIONAL, audit
                )
                roster_spec = item.get("roster_entry_list")
                if roster_spec:
                    roster_row = _record(rows, roster_spec["url"], str(roster_spec["document_id"]))
                    roster_artifact = fetch(roster_spec["url"])
                    roster_text = inspect_pdf(root / roster_artifact["path"])
                    if roster_text["document_id"] != str(roster_spec["document_id"]):
                        raise ValueError("entry-list document number differs from registry")
                    roster = parse_roster_text(
                        roster_text["text"], event, DRIVER_ALIASES, CONSTRUCTOR_ALIASES
                    )
                    roster_proof = _proof(
                        roster_spec,
                        roster_row,
                        roster_artifact,
                        roster,
                        FiaDocumentStatus.FINAL,
                        audit,
                    )
                else:
                    roster_artifact, roster_row, roster_spec = q_artifact, q_row, q_spec
                    roster = pa.Table.from_pylist(
                        [
                            {
                                "event_id": event.partition(),
                                "driver_id": row["driver_id"],
                                "constructor_id": row["constructor_id"],
                                "grid_position": None,
                                "start_type": "unknown",
                                "grid_status": "unknown",
                            }
                            for row in qualifying.to_pylist()
                        ],
                        schema=ROSTER_SCHEMA,
                    )
                    roster_proof = _proof(
                        q_spec, q_row, q_artifact, roster, FiaDocumentStatus.PROVISIONAL, audit
                    )
                if roster.num_rows != item["expected_roster_size"]:
                    raise ValueError("exact roster is incomplete")
                roster_at_cutoff(
                    rows,
                    roster_row,
                    cutoff,
                    from_qualifying=not bool(roster_spec and item.get("roster_entry_list")),
                )
                schedule = fetch(item["schedule_reference"]["url"])
                verified_schedule = validate_f1_schedule(
                    (root / schedule["path"]).read_bytes().decode("utf-8"),
                    season=event.season,
                    round_number=event.round,
                    event_name=item["event_name"],
                    circuit_id=item["circuit_id"],
                    claimed_publication=datetime.fromisoformat(
                        item["schedule_reference"]["published_at_utc"]
                    ),
                    claimed_race_start=datetime.fromisoformat(item["race_start"]),
                    source_url=schedule["url"],
                    prediction_timestamp=cutoff,
                )
                if verified_schedule.available_by > q_proof.available_at:
                    raise ValueError("scheduled context needs a later availability bound")
                audit = (
                    f"fia-winter-direct-v2:{registry['sha256']}:"
                    f"{q_artifact['sha256']}:{schedule['sha256']}"
                )
                q_proof = _proof(
                    q_spec, q_row, q_artifact, qualifying, FiaDocumentStatus.PROVISIONAL, audit
                )
                roster_proof = _proof(
                    roster_spec,
                    roster_row,
                    roster_artifact,
                    roster,
                    FiaDocumentStatus.PROVISIONAL,
                    audit,
                )
                context = PreRaceEvent(
                    event,
                    item["circuit_id"],
                    verified_schedule.race_start,
                    q_proof.available_at,
                    q_proof.available_at,
                    audit,
                )
                assert context.qualifying_completed_at is not None
                event_proof = _proof(
                    q_spec,
                    q_row,
                    q_artifact,
                    context.as_table(),
                    FiaDocumentStatus.PROVISIONAL,
                    audit,
                )
                directory = (
                    root / "data/normalized/fia_audit" / event.partition() / q_artifact["sha256"]
                )

                def table_ref(
                    table: pa.Table, proof: Any, kind: str, *, target_directory: Path = directory
                ) -> dict[str, Any]:
                    path = target_directory / f"{kind}-{table_hash(table)}.parquet"
                    if not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pq.write_table(table, path)
                    if not pq.ParquetFile(path).read().equals(table):
                        raise ValueError("retained normalized audit table changed")
                    return {
                        "kind": kind,
                        "path": path.relative_to(root).as_posix(),
                        "sha256": file_sha256(path),
                        "available_at": proof.available_at.isoformat(),
                        "evidence_reference": proof.reference,
                        "evidence": proof.to_dict(),
                    }

                def binding(
                    spec: dict[str, Any],
                    artifact: dict[str, Any],
                    proofs: list[Any],
                    *,
                    registry_reference: dict[str, Any] = registry,
                    event_identity: EventId = event,
                    audit_reference: str = audit,
                ) -> dict[str, Any]:
                    return {
                        "path": artifact["path"],
                        "sha256": artifact["sha256"],
                        "registry_path": registry_reference["path"],
                        "registry_sha256": registry_reference["sha256"],
                        "document_id": str(spec["document_id"]),
                        "document_url": spec["url"],
                        "event_id": event_identity.partition(),
                        "status": "provisional",
                        "version_audited": True,
                        "latest_at_cutoff_audited": True,
                        "audit_reference": audit_reference,
                        "table_bindings": [
                            {
                                "reference": proof.reference,
                                "table_sha256": proof.artifact_sha256,
                                "available_at_utc": proof.available_at.isoformat(),
                            }
                            for proof in proofs
                        ],
                    }

                bindings = [binding(q_spec, q_artifact, [q_proof, event_proof])]
                bindings[0]["supporting_artifacts"] = [schedule]
                if review:
                    bindings[0]["qualifying_transcription"] = review
                bindings[0]["schedule_validation"] = {
                    "publication_at_utc": verified_schedule.publication_at.isoformat(),
                    "available_by_utc": verified_schedule.available_by.isoformat(),
                    "race_start_utc": verified_schedule.race_start.isoformat(),
                    "source_row": list(verified_schedule.source_row),
                    "timezone_basis": verified_schedule.timezone_basis,
                    "html_sha256": verified_schedule.html_sha256,
                }
                if roster_artifact == q_artifact:
                    bindings[0]["table_bindings"].append(
                        {
                            "reference": roster_proof.reference,
                            "table_sha256": roster_proof.artifact_sha256,
                            "available_at_utc": roster_proof.available_at.isoformat(),
                        }
                    )
                else:
                    bindings.append(binding(roster_spec, roster_artifact, [roster_proof]))
                request = {
                    "version": 2,
                    "prediction_timestamp": cutoff.isoformat(),
                    "event": {
                        "season": event.season,
                        "round": event.round,
                        "circuit_id": item["circuit_id"],
                        "race_start": context.race_start.isoformat(),
                        "qualifying_completed_at": context.qualifying_completed_at.isoformat(),
                        "available_at": context.available_at.isoformat(),
                        "evidence_reference": event_proof.reference,
                        "evidence": event_proof.to_dict(),
                    },
                    "rosters": [table_ref(roster, roster_proof, "roster")],
                    "qualifying": [table_ref(qualifying, q_proof, "qualifying")],
                    "document_bindings": bindings,
                }
                target_spec = item["final_race_label"]
                target_row = latest_final_record(
                    rows, target_spec["url"], str(target_spec["document_id"])
                )
                label_available = _publication(target_row, winter_required=False) + timedelta(
                    minutes=1
                )
                target_artifact = fetch(target_spec["url"])
                target_text = inspect_pdf(root / target_artifact["path"])
                if target_text["document_id"] != str(target_spec["document_id"]):
                    raise ValueError(
                        "final race document identity differs from exact registry version"
                    )
                parsed = parse_final_text(
                    target_text["text"], event, DRIVER_ALIASES, CONSTRUCTOR_ALIASES
                )
                target_audit = (
                    f"fia-final-direct-v2:{registry['sha256']}:{target_artifact['sha256']}"
                )
                labels = pa.Table.from_pylist(
                    [
                        {
                            "season": event.season,
                            "round": event.round,
                            "driver_id": row["driver_id"],
                            "position": row["position"],
                            "classified": row["classified"],
                            "winner": row["classified"] and row["position"] == 1,
                            "podium": row["classified"]
                            and row["position"] is not None
                            and row["position"] <= 3,
                            "dnf": row["dnf"],
                            "dnf_category": row["dnf_category"],
                            "raw_status": row["raw_status"],
                            "taxonomy_version": DNF_TAXONOMY_VERSION,
                            "final_audited": True,
                            "label_available_at": label_available,
                            "audit_reference": target_audit,
                        }
                        for row in parsed.to_pylist()
                    ],
                    schema=OUTCOME_SCHEMA,
                )
                label_path = directory / f"outcomes-{table_hash(labels)}.parquet"
                if not label_path.exists():
                    pq.write_table(labels, label_path)
                if not pq.ParquetFile(label_path).read().equals(labels):
                    raise ValueError("retained audited outcome table changed")
                request["outcomes"] = {
                    "path": label_path.relative_to(root).as_posix(),
                    "sha256": file_sha256(label_path),
                }
                request["outcome_document_bindings"] = [
                    {
                        "path": target_artifact["path"],
                        "sha256": target_artifact["sha256"],
                        "document_id": str(target_spec["document_id"]),
                        "document_url": target_spec["url"],
                        "event_id": event.partition(),
                        "status": "final",
                        "version_audited": True,
                        "latest_final_audited": True,
                        "audit_reference": target_audit,
                        "outcome_sha256": file_sha256(label_path),
                        "label_available_at_utc": label_available.isoformat(),
                        "raw_publication_cet": target_row["publication_cet"],
                        "label_clock_policy": "conservative UTC+1 upper bound; minute end",
                        "publication_timezone_status": "unresolved_CET_CEST"
                        if datetime.strptime(target_row["publication_cet"], "%d.%m.%y %H:%M")
                        .replace(tzinfo=ZoneInfo("Europe/Paris"))
                        .utcoffset()
                        != timedelta(hours=1)
                        else "verified_literal_CET",
                        "registry_path": registry["path"],
                        "registry_sha256": registry["sha256"],
                    }
                ]
                request_path = (
                    directory / f"request-{hashlib.sha256(_json(request)).hexdigest()}.json"
                )
                _immutable(request_path, _json(request))
                outcome = reconstruct_gold_core(request_path, root)
                result.update(
                    {
                        "status": "included",
                        "snapshot_id": outcome["snapshot_id"],
                        "rows": roster.num_rows,
                        "request": request_path.relative_to(root).as_posix(),
                        "evidence_manifest": outcome["evidence_manifest"],
                        "reasons": [],
                    }
                )
            except (ValueError, RuntimeError, OSError, KeyError, httpx.HTTPError) as exc:
                result["reasons"] = [str(exc)]
            results.append(result)
            if sum(row["status"] == "included" for row in results) >= minimum_races:
                break
    finally:
        if own:
            client.close()
    report = {
        "version": 1,
        "audit_method": "fia-winter-direct-v2",
        "tier": "Gold",
        "minimum_gold_races": minimum_races,
        "included_races": sum(row["status"] == "included" for row in results),
        "races": results,
        "catalog_sha256": file_sha256(catalog_path),
        "unattempted_races": [
            {
                "event_id": EventId(item["season"], item["round"]).partition(),
                "reason": "sufficient_audited_coverage_reached",
            }
            for item in candidates[len(results) :]
        ],
    }
    report["status"] = (
        "eligible" if report["included_races"] >= minimum_races else "insufficient_data"
    )
    path = (
        root
        / "data/benchmarks/historical_audit"
        / f"winter-{hashlib.sha256(_json(report)).hexdigest()}.json"
    )
    _immutable(path, _json(report))
    return report
