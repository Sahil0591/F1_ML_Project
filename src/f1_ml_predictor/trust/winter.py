"""Direct FIA winter-publication audits over a bounded researched candidate pool."""

import hashlib
import json
import re
import time
import unicodedata
from calendar import month_name
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
from f1_ml_predictor.trust.f1_schedule import (
    validate_event_timetable,
    validate_f1_schedule,
    validate_fia_timetable_amendment,
    validate_race_time_article,
)
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
    qualifying_car_numbers,
)
from f1_ml_predictor.trust.historical import (
    _immutable,
    _json,
    _official,
    _retain_response,
    inspect_pdf,
    reconstruct_gold_core,
    verify_post_final_review,
)
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION, OUTCOME_SCHEMA
from f1_ml_predictor.trust.transcription import reviewed_image_final, reviewed_qualifying

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
    "Kevin MAGNUSSEN": "magnussen",
    "Mick SCHUMACHER": "mick_schumacher",
    "Daniel RICCIARDO": "ricciardo",
    "Sebastian VETTEL": "vettel",
    "Zhou GUANYU": "zhou",
    "Guanyu ZHOU": "zhou",
    "ZHOU Guanyu": "zhou",
    "Nicholas LATIFI": "latifi",
    "Nyck DE VRIES": "de_vries",
    "Logan SARGEANT": "sargeant",
    "Arvid LINDBLAD": "lindblad",
}
_CONSTRUCTORS = {
    "Oracle Red Bull Racing": "red_bull",
    "Red Bull Racing": "red_bull",
    "McLaren Formula 1 Team": "mclaren",
    "McLaren F1 Team": "mclaren",
    "McLaren Mastercard F1 Team": "mclaren",
    "McLaren": "mclaren",
    "Mercedes-AMG PETRONAS F1 Team": "mercedes",
    "Mercedes-AMG Petronas F1 Team": "mercedes",
    "Mercedes AMG-PETRONAS F1 Team": "mercedes",
    "Mercedes": "mercedes",
    "Scuderia Ferrari HP": "ferrari",
    "Scuderia Ferrari": "ferrari",
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
    "Haas F1 Team": "haas",
    "Alfa Romeo F1 Team ORLEN": "alfa_romeo",
    "Alfa Romeo F1 Team Stake": "alfa_romeo",
    "Alfa Romeo F1 Team Kick": "alfa_romeo",
    "Scuderia AlphaTauri": "alpha_tauri",
    "Aston Martin Aramco Cognizant F1 Team": "aston_martin",
    "Aston Martin Aramco Cognizant F1": "aston_martin",
    "Williams Racing": "williams",
    "TGR Haas F1 Team": "haas",
    "Haas Ferrari": "haas",
    "Visa Cash App Racing Bulls F1 Team": "rb",
    "Visa Cash App R acing Bulls F1 Team": "rb",
    "Racing Bulls Honda RBPT": "rb",
    "Visa Cash App RB Formula One Team": "rb",
    "Visa Cash App RB F1 Team": "rb",
    "Racing Bulls Red Bull Ford": "rb",
    "Kick Sauber F1 Team": "sauber",
    "Stake F1 Team Kick Sauber": "sauber",
    "Sauber": "sauber",
    "Audi Revolut F1 Team": "audi",
    "Audi": "audi",
    "Cadillac Formula 1 Team": "cadillac",
    # Miami 2026 Doc 12 merges the team and chassis columns for car 11.
    "Cadillac Formula 1Team Cadillac Ferrari": "cadillac",
    "Cadillac Ferrari": "cadillac",
    "Red Bull Racing Honda RBPT": "red_bull",
    "Red Bull Racing Red Bull Ford": "red_bull",
    "McLaren Mercedes": "mclaren",
    "Aston Martin Aramco Honda": "aston_martin",
}
DRIVER_ALIASES = {**_DRIVERS, **{key.title(): value for key, value in _DRIVERS.items()}}
CONSTRUCTOR_ALIASES = _CONSTRUCTORS
_CONSTRUCTOR_ALIASES_BY_SEASON = {
    2024: {
        "RB Honda RBPT": "rb",
        "Kick Sauber Ferrari": "sauber",
    },
    2025: {
        "Kick Sauber Ferrari": "sauber",
    },
}


def constructor_aliases_for_season(season: int) -> dict[str, str]:
    """Use only contemporary, unambiguous FIA constructor name variants."""
    return {**CONSTRUCTOR_ALIASES, **_CONSTRUCTOR_ALIASES_BY_SEASON.get(season, {})}


def registry_rows(text: str) -> list[dict[str, Any]]:
    """Keep recalled rows as well as downloadable document publication records."""
    rows = []
    for block in re.findall(r'<li\b[^>]*class="document-row[^>]*>.*?</li>', text, re.S):
        plain = " ".join(unescape(re.sub(r"<[^>]+>", " ", block)).split())
        match = re.search(
            r"^(?:Recalled\s*-\s*)?(?:Doc\s+(\d+)\s*-\s*)?(.*?)\s+Published on\s+"
            r"(\d{2}\.\d{2}\.\d{2}\s+\d{2}:\d{2})\s+CET(?:\s|$)",
            plain,
        )
        if not match:
            continue
        url = re.search(r'<a\b[^>]*href="([^"]+)"', block)
        if not match[1] and url is None:
            continue
        rows.append(
            {
                # Older registry pages omit the document number. The PDF cover
                # must supply it before a direct evidence binding is certified.
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


def _publication(row: dict[str, Any]) -> datetime:
    """Use the later UTC bound if the registry's CET label means CET or CEST."""
    local = datetime.strptime(row["publication_cet"], "%d.%m.%y %H:%M")
    return local.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC)


def _version_key(row: dict[str, Any]) -> tuple[datetime, int, str]:
    return _publication(row), int(row["document_id"] or 0), row.get("url") or ""


def _record(rows: list[dict[str, Any]], url: str, identifier: str | None) -> dict[str, Any]:
    _official(url)
    matching = [
        row
        for row in rows
        if row["url"] == url and (identifier is None or row["document_id"] in {None, identifier})
    ]
    if len(matching) != 1 or matching[0]["recalled"]:
        raise ValueError("exact downloadable document version is missing, recalled or ambiguous")
    if identifier is None and matching[0]["document_id"] is not None:
        raise ValueError("document number is absent from the research catalog")
    return matching[0]


def _bind_pdf_identity(
    row: dict[str, Any],
    spec: dict[str, Any],
    inspected: dict[str, Any],
    item: dict[str, Any],
    *,
    require_title: bool = True,
) -> None:
    """Fill legacy registry numbers only from the retained matching PDF cover."""
    number = inspected["document_id"]
    if number is None or (row["document_id"] is not None and row["document_id"] != number):
        raise ValueError("downloaded PDF document number contradicts registry")
    if spec["document_id"] is not None and str(spec["document_id"]) != number:
        raise ValueError("downloaded PDF document number contradicts research catalog")
    if row["document_id"] is not None:
        spec["document_id"] = number
        return

    def words(value: str) -> str:
        ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
        return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))

    cover = words(inspected["cover_text"])
    event = words(item.get("fia_event_name", item["event_name"]))
    title = words(row["title"])
    date_match = re.search(
        r"\bdate (\d{1,2}) (" + "|".join(name.lower() for name in month_name[1:]) + r") (\d{4})\b",
        cover,
    )
    published_day = datetime.strptime(row["publication_cet"], "%d.%m.%y %H:%M").date()
    issued_day = (
        datetime(
            int(date_match[3]), list(month_name).index(date_match[2].title()), int(date_match[1])
        ).date()
        if date_match
        else None
    )
    if (
        str(item["season"]) not in cover.split()
        or event not in cover
        or (require_title and title not in cover)
        or issued_day is None
        or abs((issued_day - published_day).days) > 1
    ):
        raise ValueError("PDF cover contradicts season, event, date, or document type")
    row["document_id"] = number
    spec["document_id"] = number


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
        or max(known, key=_version_key) != selected
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
    latest = max(known, key=_version_key)
    if from_qualifying:
        if _publication(latest) >= _publication(selected):
            raise ValueError("newer entry-list state requires roster review")
    elif (
        latest != selected
        or selected["recalled"]
        or _publication(selected) + timedelta(minutes=1) > cutoff
    ):
        raise ValueError("selected roster is not the latest known entry-list state")


def latest_final_record(
    rows: list[dict[str, Any]],
    url: str,
    identifier: str | None,
    *,
    review: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
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
            key=_version_key,
        )
        != selected
    ):
        raise ValueError("selected final target is not the latest official classification")
    # The retained registry is already scoped to one event. FIA replacement
    # filenames can use another case, delimiter, or hosting path.
    later = [
        row
        for row in rows
        if not row["recalled"]
        and row.get("url")
        and _version_key(row) > _version_key(selected)
        and row["title"].lower() != "championship points"
    ]
    if later:
        if review is None or root is None:
            raise ValueError("later event documents require final outcome review")
        expected = {
            (row["document_id"], row["title"], row["url"]) for row in review["later_documents"]
        }
        observed = {(row["document_id"], row["title"], row["url"]) for row in later}
        if expected != observed:
            raise ValueError("later event documents differ from the completed review")
        conclusion = review.get("conclusion")
        if conclusion in {
            "media_procedure_only",
            "no_penalty_applied",
            "no_race_classification_change",
        }:
            if (
                conclusion == "no_race_classification_change"
                and review.get("final_publication_cet") != selected["publication_cet"]
            ):
                raise ValueError("post-final review names another final publication")
            verify_post_final_review(review, root)
            return selected
        if conclusion != "classification_cannot_be_amended":
            raise ValueError("later event documents differ from the completed review")
        decisions = [row for row in later if row["title"].startswith("Decision - Williams")]
        if len(decisions) != 1 or decisions[0]["document_id"] != review["decision_document_id"]:
            raise ValueError("review decision identity changed")
        artifact = review["decision_artifact"]
        document = _safe_file(root, artifact["path"], artifact["sha256"])
        inspected = inspect_pdf(document)
        normalized = " ".join(inspected["text"].lower().split())
        if (
            inspected["document_id"] != decisions[0]["document_id"]
            or "no power to remedy that served time penalty by amending the classifications"
            not in normalized
        ):
            raise ValueError("review does not establish unchanged final classification")
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
        not binding["audit_reference"].startswith(("fia-winter-direct-v2:", "fia-direct-v3:"))
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
    exhaustive: bool = False,
) -> dict[str, Any]:
    """Audit exact document values, optionally visiting the entire catalog.

    No current API classifications are used. Independent source downloads remain
    current-state artifacts; only the checked exact document/table association is
    assigned direct audited publication evidence. Optional features stay missing.
    """
    root = root.resolve()
    candidates = json.loads(catalog_path.read_text(encoding="utf-8"))["candidates"]
    image_reviews = json.loads(
        (root / "docs/HISTORICAL_IMAGE_FINAL_REVIEWS.json").read_text(encoding="utf-8")
    )["reviews"]
    if not 8 <= minimum_races <= 40 or not 1 <= len(candidates) <= (200 if exhaustive else 60):
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
                    if (
                        not exhaustive
                        and sum(row["status"] == "included" for row in results) >= minimum_races
                    ):
                        break
                    continue
                _official(item["index_url"])
                if item.get("schedule_reference") is None:
                    raise ValueError("cutoff_valid_official_race_schedule_missing")
                if item.get("final_race_label") is None:
                    raise ValueError("final_race_classification_missing")
                registry = fetch(item["index_url"])
                rows = registry_rows((root / registry["path"]).read_text(encoding="utf-8"))
                q_spec = {"url": item["qualifying_url"], "document_id": item["document_id"]}
                if "\ufffd" in q_spec["url"]:
                    resolved = [
                        row
                        for row in rows
                        if row.get("url")
                        and not row["recalled"]
                        and row["document_id"] == q_spec["document_id"]
                        and "qualifying classification" in row["title"].lower()
                        and "sprint" not in row["title"].lower()
                        and _publication(row).isoformat() == item["published_at_utc"]
                    ]
                    if len(resolved) != 1:
                        raise ValueError("qualifying URL encoding cannot be resolved exactly")
                    q_spec["url"] = resolved[0]["url"]
                q_row = _record(rows, q_spec["url"], q_spec["document_id"])
                published = _publication(q_row)
                if published != datetime.fromisoformat(item["published_at_utc"]):
                    raise ValueError("research publication claim differs from retained registry")
                cutoff = datetime.fromisoformat(item["prediction_timestamp_utc"])
                constructors = constructor_aliases_for_season(event.season)
                q_artifact = fetch(q_spec["url"])
                q_text = inspect_pdf(root / q_artifact["path"])
                _bind_pdf_identity(q_row, q_spec, q_text, item)
                qualifying_at_cutoff(rows, q_row, cutoff)
                review = item.get("qualifying_transcription")
                qualifying = (
                    reviewed_qualifying(
                        q_text["text"],
                        event,
                        DRIVER_ALIASES,
                        constructors,
                        review,
                        q_artifact["sha256"],
                    )
                    if review
                    else parse_qualifying_text(q_text["text"], event, DRIVER_ALIASES, constructors)
                )
                audit = f"fia-winter-direct-v1:{registry['sha256']}:{q_artifact['sha256']}"
                q_proof = _proof(
                    q_spec, q_row, q_artifact, qualifying, FiaDocumentStatus.PROVISIONAL, audit
                )
                roster_spec = item.get("roster_entry_list")
                if not roster_spec and qualifying.num_rows < item["expected_roster_size"]:
                    entry_lists = [
                        row
                        for row in rows
                        if "entry list" in row["title"].lower()
                        and not row["recalled"]
                        and _publication(row) + timedelta(minutes=1) <= cutoff
                    ]
                    if entry_lists:
                        selected_entry = max(
                            entry_lists,
                            key=_version_key,
                        )
                        roster_spec = {
                            "url": selected_entry["url"],
                            "document_id": selected_entry["document_id"],
                        }
                if roster_spec:
                    roster_row = _record(rows, roster_spec["url"], roster_spec["document_id"])
                    roster_artifact = fetch(roster_spec["url"])
                    roster_text = inspect_pdf(root / roster_artifact["path"])
                    _bind_pdf_identity(roster_row, roster_spec, roster_text, item)
                    roster = parse_roster_text(
                        roster_text["text"], event, DRIVER_ALIASES, constructors
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
                pre_cutoff_withdrawals = []
                if roster_artifact != q_artifact and qualifying.num_rows < roster.num_rows:
                    qualifying_ids = {row["driver_id"] for row in qualifying.to_pylist()}
                    entry_rows = {row["driver_id"]: row for row in roster.to_pylist()}
                    absent = set(entry_rows) - qualifying_ids
                    if qualifying_ids <= set(entry_rows) and absent:
                        for driver_id in sorted(absent):
                            displays = [
                                name
                                for name, identity in DRIVER_ALIASES.items()
                                if identity == driver_id
                            ]
                            numbers = {
                                int(match[1])
                                for line in roster_text["text"].splitlines()
                                for name in displays
                                if (match := re.match(rf"^\s*(\d+)\s+{re.escape(name)}\b", line))
                            }
                            if len(numbers) != 1:
                                break
                            car_number = numbers.pop()
                            decisions = [
                                row
                                for row in rows
                                if row.get("url")
                                and not row["recalled"]
                                and row["title"].lower().startswith(("information", "decision"))
                                and _publication(row) + timedelta(minutes=1) <= cutoff
                            ]
                            matched = []
                            for decision in decisions:
                                decision_artifact = fetch(decision["url"])
                                decision_pdf = inspect_pdf(root / decision_artifact["path"])
                                plain = " ".join(
                                    re.findall(r"[a-z0-9]+", decision_pdf["text"].lower())
                                )
                                if any(
                                    re.search(
                                        rf"withdrawing\s+car\s+{car_number}\s+driver\s+"
                                        + r"\s+".join(
                                            re.escape(part) for part in name.lower().split()
                                        ),
                                        plain,
                                    )
                                    for name in displays
                                ):
                                    matched.append((decision, decision_artifact, decision_pdf))
                            if len(matched) != 1:
                                break
                            decision, decision_artifact, decision_pdf = matched[0]
                            decision_spec = {
                                "url": decision["url"],
                                "document_id": decision["document_id"],
                            }
                            _bind_pdf_identity(
                                decision, decision_spec, decision_pdf, item, require_title=False
                            )
                            pre_cutoff_withdrawals.append(
                                {
                                    "driver_id": driver_id,
                                    "car_number": car_number,
                                    "document_id": decision_spec["document_id"],
                                    "url": decision["url"],
                                    "publication_cet": decision["publication_cet"],
                                    "available_at_utc": (
                                        _publication(decision) + timedelta(minutes=1)
                                    ).isoformat(),
                                    "artifact": decision_artifact,
                                }
                            )
                        if len(pre_cutoff_withdrawals) == len(absent):
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
                                q_spec,
                                q_row,
                                q_artifact,
                                roster,
                                FiaDocumentStatus.PROVISIONAL,
                                audit,
                            )
                        else:
                            pre_cutoff_withdrawals = []
                expected_size = item["expected_roster_size"] - len(pre_cutoff_withdrawals)
                if roster.num_rows != expected_size:
                    raise ValueError("exact roster is incomplete")
                roster_at_cutoff(
                    rows,
                    roster_row,
                    cutoff,
                    from_qualifying=roster_artifact == q_artifact,
                )
                schedule_kind = item["schedule_reference"].get("kind")
                schedule_url = item["schedule_reference"]["url"]
                if schedule_kind == "fia_timetable_amendment":
                    amendment_rows = [
                        row
                        for row in rows
                        if row["document_id"] == item["schedule_reference"]["document_id"]
                        and "change to timetable" in row["title"].lower()
                        and _publication(row).isoformat()
                        == item["schedule_reference"]["published_at_utc"]
                        and not row["recalled"]
                        and row.get("url")
                    ]
                    if len(amendment_rows) != 1:
                        raise ValueError("FIA timetable amendment registry identity is ambiguous")
                    schedule_url = amendment_rows[0]["url"]
                schedule = fetch(schedule_url)
                schedule_claim = {
                    "season": event.season,
                    "round_number": event.round,
                    "event_name": item["event_name"],
                    "circuit_id": item["circuit_id"],
                    "claimed_publication": datetime.fromisoformat(
                        item["schedule_reference"]["published_at_utc"]
                    ),
                    "claimed_race_start": datetime.fromisoformat(item["race_start"]),
                    "source_url": schedule_url,
                    "prediction_timestamp": cutoff,
                }
                if schedule_kind == "fia_timetable_amendment":
                    schedule_pdf = inspect_pdf(root / schedule["path"])
                    schedule_row = _record(
                        rows, schedule_url, item["schedule_reference"]["document_id"]
                    )
                    if schedule_pdf["document_id"] != str(schedule_row["document_id"]):
                        raise ValueError("FIA timetable amendment document identity changed")
                    verified_schedule = validate_fia_timetable_amendment(
                        schedule_pdf["text"],
                        **schedule_claim,
                        document_sha256=schedule["sha256"],
                    )
                else:
                    schedule_validator = (
                        validate_event_timetable
                        if schedule_kind == "event_timetable"
                        else validate_race_time_article
                        if schedule_kind == "race_time_article"
                        else validate_f1_schedule
                    )
                    verified_schedule = schedule_validator(
                        (root / schedule["path"]).read_bytes().decode("utf-8"),
                        **schedule_claim,
                    )
                if verified_schedule.available_by > q_proof.available_at:
                    raise ValueError("scheduled context needs a later availability bound")
                audit = (
                    f"fia-direct-v3:cet-upper-bound:{registry['sha256']}:"
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
                    stored = pq.ParquetFile(path).read()
                    if not stored.schema.equals(table.schema) or table_hash(stored) != table_hash(
                        table
                    ):
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
                if pre_cutoff_withdrawals:
                    bindings[0]["pre_cutoff_withdrawals"] = pre_cutoff_withdrawals
                qualifying_chain = [
                    row
                    for row in rows
                    if "qualifying classification" in row["title"].lower()
                    and "sprint" not in row["title"].lower()
                    and _publication(row) <= cutoff
                ]
                if any(row["recalled"] for row in qualifying_chain):
                    bindings[0]["qualifying_document_chain"] = [
                        {
                            "document_id": row["document_id"],
                            "title": row["title"],
                            "publication_cet": row["publication_cet"],
                            "url": row["url"],
                            "recalled": row["recalled"],
                            "selected": row == q_row,
                        }
                        for row in sorted(qualifying_chain, key=_version_key)
                    ]
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
                if "\ufffd" in target_spec["url"]:
                    resolved = [
                        row
                        for row in rows
                        if row.get("url")
                        and not row["recalled"]
                        and row["document_id"] == target_spec["document_id"]
                        and row["title"].lower() == "final race classification"
                    ]
                    if len(resolved) != 1:
                        raise ValueError("final URL encoding cannot be resolved exactly")
                    target_spec = {**target_spec, "url": resolved[0]["url"]}
                target_row = latest_final_record(
                    rows,
                    target_spec["url"],
                    target_spec["document_id"],
                    review=item.get("post_final_review"),
                    root=root,
                )
                label_available = _publication(target_row) + timedelta(minutes=1)
                target_artifact = fetch(target_spec["url"])
                target_text = inspect_pdf(root / target_artifact["path"])
                _bind_pdf_identity(target_row, target_spec, target_text, item)
                image_review = image_reviews.get(event.partition())
                parsed = (
                    reviewed_image_final(
                        target_text["text"],
                        event,
                        DRIVER_ALIASES,
                        constructors,
                        image_review,
                        target_artifact["sha256"],
                    )
                    if image_review
                    else parse_final_text(target_text["text"], event, DRIVER_ALIASES, constructors)
                )
                roster_by_driver = {row["driver_id"]: row for row in roster.to_pylist()}
                final_by_driver = {row["driver_id"]: row for row in parsed.to_pylist()}
                if not set(final_by_driver) <= set(roster_by_driver):
                    raise ValueError(
                        "final classification includes a driver outside the cutoff roster"
                    )
                transitions = []
                if set(final_by_driver) != set(roster_by_driver):
                    car_numbers = qualifying_car_numbers(
                        q_text["text"], event, DRIVER_ALIASES, constructors
                    )
                    for driver_id in sorted(set(roster_by_driver) - set(final_by_driver)):
                        qualified_car_number = car_numbers.get(driver_id)
                        if qualified_car_number is None:
                            raise ValueError(
                                "missing final driver has no qualifying car-number proof"
                            )
                        decisions = [
                            row
                            for row in rows
                            if not row["recalled"]
                            and row.get("url")
                            and "withdrawal from the competition" in row["title"].lower()
                            and re.search(rf"\bcar\s+{qualified_car_number}\b", row["title"], re.I)
                            and cutoff < _publication(row) < _publication(target_row)
                        ]
                        if len(decisions) != 1:
                            raise ValueError(
                                "missing final driver lacks one later FIA withdrawal decision"
                            )
                        decision = decisions[0]
                        decision_artifact = fetch(decision["url"])
                        decision_spec = {
                            "url": decision["url"],
                            "document_id": decision["document_id"],
                        }
                        decision_pdf = inspect_pdf(root / decision_artifact["path"])
                        _bind_pdf_identity(decision, decision_spec, decision_pdf, item)
                        decision_text = " ".join(
                            re.findall(r"[a-z0-9]+", decision_pdf["text"].lower())
                        )
                        if (
                            not re.search(
                                rf"withdraw\s+car\s+{qualified_car_number}\s+from\s+the\s+competition",
                                decision_text,
                            )
                            or "this request is approved" not in decision_text
                        ):
                            raise ValueError(
                                "FIA decision does not approve the exact car withdrawal"
                            )
                        transitions.append(
                            {
                                "kind": "post_qualifying_withdrawal",
                                "driver_id": driver_id,
                                "car_number": qualified_car_number,
                                "document_id": str(decision_spec["document_id"]),
                                "document_url": decision_spec["url"],
                                "path": decision_artifact["path"],
                                "sha256": decision_artifact["sha256"],
                                "registry_path": registry["path"],
                                "registry_sha256": registry["sha256"],
                                "publication_cet": decision["publication_cet"],
                                "available_at_utc": (
                                    _publication(decision) + timedelta(minutes=1)
                                ).isoformat(),
                            }
                        )
                        final_by_driver[driver_id] = {
                            "event_id": event.partition(),
                            "driver_id": driver_id,
                            "constructor_id": roster_by_driver[driver_id]["constructor_id"],
                            "position": None,
                            "classified": False,
                            "raw_status": "approved post-qualifying withdrawal",
                            "dnf_category": "did_not_start",
                            "dnf": None,
                        }
                    parsed = pa.Table.from_pylist(
                        list(final_by_driver.values()), schema=parsed.schema
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
                stored_labels = pq.ParquetFile(label_path).read()
                if not stored_labels.schema.equals(labels.schema) or table_hash(
                    stored_labels
                ) != table_hash(labels):
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
                        "post_final_review": item.get("post_final_review"),
                        "roster_transitions": transitions,
                        "image_final_review": image_review,
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
            if (
                not exhaustive
                and sum(row["status"] == "included" for row in results) >= minimum_races
            ):
                break
    finally:
        if own:
            client.close()
    report = {
        "version": 1,
        "audit_method": "fia-direct-v3-cet-upper-bound",
        "tier": "Gold",
        "minimum_gold_races": minimum_races,
        "exhaustive": exhaustive,
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
