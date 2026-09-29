"""Bounded later FIA final classifications, kept separate from frozen features."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import normalize_qualifying, normalize_schedule
from f1_ml_predictor.sources.jolpica import BASE_URL
from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.collected_features import _collection
from f1_ml_predictor.trust.fia_tables import parse_final_text
from f1_ml_predictor.trust.historical import (
    _immutable,
    _json,
    _official,
    _retain_response,
    inspect_pdf,
)
from f1_ml_predictor.trust.outcomes import (
    DNF_TAXONOMY_VERSION,
    OUTCOME_SCHEMA,
    validate_audited_outcomes,
)
from f1_ml_predictor.trust.prospective import load_bundle
from f1_ml_predictor.trust.winter import (
    CONSTRUCTOR_ALIASES,
    DRIVER_ALIASES,
    _publication,
    latest_final_record,
    registry_rows,
)

_ROOT_URL = "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14"


def _clock(now: Callable[[], datetime] | None) -> datetime:
    clock = datetime.now(UTC) if now is None else now()
    if not isinstance(clock, datetime):
        raise ValueError("outcome clock must return a timezone-aware UTC datetime")
    require_utc(clock, "outcome observation clock")
    return clock


class _Options(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.groups: list[list[dict[str, str]]] = []
        self.group: list[dict[str, str]] | None = None
        self.option: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "select":
            self.group = []
        if tag == "option" and self.group is not None:
            self.option = {"url": dict(attrs).get("value") or "", "title": ""}

    def handle_data(self, data: str) -> None:
        if self.option is not None:
            self.option["title"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "option" and self.option is not None and self.group is not None:
            self.option["title"] = " ".join(self.option["title"].split())
            self.group.append(self.option)
            self.option = None
        if tag == "select" and self.group is not None:
            self.groups.append(self.group)
            self.group = None


def _options(text: str) -> list[list[dict[str, str]]]:
    parser = _Options()
    parser.feed(text)
    return parser.groups


def _exact_option(text: str, title: str) -> str:
    options = [row for group in _options(text) for row in group if row["title"] == title]
    if len(options) != 1:
        raise ValueError("official FIA season/event option is missing or ambiguous")
    url = urljoin("https://www.fia.com", options[0]["url"])
    _official(url)
    return url


def _selected_registry(text: str, season: int, race_name: str) -> None:
    groups = _options(text)
    if not any(group and group[0]["title"] == f"SEASON {season}" for group in groups):
        raise ValueError("FIA registry does not retain the exact selected season")
    if not any(group and group[0]["title"] == race_name for group in groups):
        raise ValueError("FIA registry does not retain the exact selected event")


def _registry(text: str, season: int, race_name: str) -> list[dict[str, Any]]:
    _selected_registry(text, season, race_name)
    rows = registry_rows(text)
    blocks = re.findall(r'<li\b[^>]*class="document-row[^>]*>.*?</li>', text, re.S)
    if len(rows) != len(blocks) or len({row["document_id"] for row in rows}) != len(rows):
        raise ValueError("FIA registry has unresolved publication clocks or document versions")
    return rows


def _tokens(value: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", value.casefold()).split())


def _same_event_url(url: str | None, season: int, race_name: str) -> bool:
    if not isinstance(url, str):
        return False
    filename = unquote(urlsplit(url).path).rsplit("/", 1)[-1]
    prefix = re.split(r"(?:_-_| - )", filename, maxsplit=1)[0]
    return _tokens(prefix) == _tokens(f"{season} {race_name}")


def _final(rows: list[dict[str, Any]], event: EventId, race_name: str) -> dict[str, Any]:
    finals = [
        row
        for row in rows
        if "final race classification" in row["title"].lower()
        and "sprint" not in row["title"].lower()
        and not row["recalled"]
    ]
    if not finals or any(
        not _same_event_url(row.get("url"), event.season, race_name) for row in finals
    ):
        raise ValueError(
            "latest nonrecalled final classification is missing or belongs to another event"
        )
    selected = max(finals, key=lambda row: (_publication(row), int(row["document_id"])))
    latest_final_record(rows, selected["url"], selected["document_id"])
    key = (_publication(selected), int(selected["document_id"]))
    later = [
        row
        for row in rows
        if not row["recalled"]
        and (_publication(row), int(row["document_id"])) > key
        and row["title"].lower() != "championship points"
    ]
    if later:
        raise ValueError("later same-event rulings require a newer final classification")
    return selected


def _retained_context(
    bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    manifest, tables = load_bundle(bundle)
    event = EventId(**manifest["event"])
    requests = manifest["request_metadata"]["source_requests"]
    role_names = {
        role: [name for name, request in requests.items() if request.get("role") == role]
        for role in ("event", "qualifying")
    }
    if any(len(names) != 1 for names in role_names.values()):
        raise ValueError("later outcomes require exact retained schedule and qualifying roles")

    def races(name: str) -> list[dict[str, Any]]:
        if tables[name].column_names != ["payload_json"] or tables[name].num_rows != 1:
            raise ValueError("later outcomes require one retained JSON payload per role")
        payload = json.loads(tables[name]["payload_json"][0].as_py())
        return _collection(payload, qualifying=name == role_names["qualifying"][0])

    expected = {
        role_names["event"][0]: f"{BASE_URL}/{event.season}",
        role_names["qualifying"][0]: f"{BASE_URL}/{event.season}/{event.round}/qualifying",
    }
    if any(requests[name].get("url", "").rstrip("/") != url for name, url in expected.items()):
        raise ValueError("later outcomes require exact captured Jolpica event endpoints")

    schedule = races(role_names["event"][0])
    normalized = normalize_schedule(schedule, event.season)
    selected = [row for row in normalized.to_pylist() if row["event_id"] == event.partition()]
    if len(selected) != 1:
        raise ValueError("retained schedule is missing the exact outcome event")
    raw = next(row for row in schedule if int(row["round"]) == event.round)
    qualifying_races = races(role_names["qualifying"][0])
    normalized_q = normalize_qualifying(qualifying_races, event)
    if not normalized_q.num_rows:
        raise ValueError("outcome collection requires a nonempty retained qualifying roster")
    constructors = {row["driver_id"]: row["constructor_id"] for row in normalized_q.to_pylist()}
    if len(constructors) != normalized_q.num_rows:
        raise ValueError("later outcomes require unique captured qualifying driver identities")
    raw_entries = [entry for row in qualifying_races for entry in row["QualifyingResults"]]
    return manifest, raw, raw_entries, constructors


def collect_final_outcomes(
    root: Path,
    bundle: Path,
    *,
    http_client: httpx.Client | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Capture one current final document after all visible same-event rulings.

    At most five official requests select a season, exact event, registry, PDF,
    and registry recheck. Missing versions, aliases, clocks, or roster matches
    raise and can retry later. Printed classification alone supplies no DNF cause.
    """
    root, bundle = root.resolve(), bundle.resolve()
    if not bundle.is_relative_to(root):
        raise ValueError("outcome capture must remain within the project root")
    manifest, raw_race, raw_entries, constructors = _retained_context(bundle)
    event = EventId(**manifest["event"])
    race_start = datetime.fromisoformat(manifest["request_metadata"]["window"]["race_start"])
    require_utc(race_start, "retained race start")
    if _clock(now) <= race_start:
        raise ValueError("automatic audited labels must be collected after race start")
    normalized_race = normalize_schedule([raw_race], event.season).to_pylist()[0]
    if normalized_race["race_start_utc"] != race_start:
        raise ValueError("retained outcome schedule disagrees with frozen prediction window")
    race_name = normalized_race["race_name"]
    drivers = dict(DRIVER_ALIASES)
    teams = dict(CONSTRUCTOR_ALIASES)
    for entry in raw_entries:
        driver, constructor = entry["Driver"], entry["Constructor"]
        if driver.get("givenName") and driver.get("familyName"):
            for display in (
                f"{driver['givenName']} {driver['familyName']}",
                f"{driver['givenName']} {driver['familyName'].upper()}",
            ):
                if display in drivers and drivers[display] != driver["driverId"]:
                    raise ValueError("captured driver identity conflicts with exact FIA alias")
                drivers[display] = driver["driverId"]
        if constructor.get("name"):
            display = constructor["name"]
            if display in teams and teams[display] != constructor["constructorId"]:
                raise ValueError("captured constructor identity conflicts with exact FIA alias")
            teams[display] = constructor["constructorId"]
    client = http_client or httpx.Client(timeout=10, follow_redirects=False)
    own = http_client is None
    observations: list[dict[str, Any]] = []

    def fetch(url: str) -> tuple[dict[str, Any], httpx.Response]:
        _official(url)
        if len(observations) >= 5:
            raise ValueError("bounded outcome request budget exhausted")
        response = client.get(
            url, timeout=10, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
        )
        if str(response.url) != url:
            raise ValueError("automatic FIA collection cannot follow an unverified redirect")
        artifact = _retain_response(root, response)
        artifact = {**artifact, "observed_at_utc": _clock(now).isoformat()}
        observations.append(artifact)
        return artifact, response

    try:
        _, root_response = fetch(_ROOT_URL)
        season_url = _exact_option(root_response.text, f"SEASON {event.season}")
        if not re.fullmatch(
            re.escape(_ROOT_URL) + rf"/season/season-{event.season}-[0-9]+", season_url
        ):
            raise ValueError("discovered FIA season URL does not match the retained event year")
        _, season_response = fetch(season_url)
        registry_url = _exact_option(season_response.text, race_name)
        if (
            not registry_url.startswith(season_url + "/event/")
            or unquote(registry_url[len(season_url + "/event/") :]) != race_name
        ):
            raise ValueError("discovered FIA registry URL does not match the exact retained event")
        registry_artifact, registry_response = fetch(registry_url)
        selected = _final(
            _registry(registry_response.text, event.season, race_name), event, race_name
        )
        final_artifact, final_response = fetch(selected["url"])
        if not final_response.content.startswith(b"%PDF"):
            raise ValueError("final race classification response is not a PDF")
        try:
            inspected = inspect_pdf(root / final_artifact["path"])
        except Exception as exc:
            # PDF libraries expose several parser-specific errors. A malformed
            # official response must remain retryable and cannot stop the tick.
            raise ValueError("final FIA PDF could not be inspected") from exc
        if inspected["document_id"] != selected["document_id"]:
            raise ValueError("final PDF document number differs from the exact registered version")
        text = _tokens(inspected.get("cover_text", "") + "\n" + inspected["text"])
        names = [race_name, race_name.removesuffix("Grand Prix") + "GP"]
        if (
            not any(
                _tokens(f"{event.season} {name}") in text
                or _tokens(f"{name} {event.season}") in text
                for name in names
            )
            or "final race classification" not in text
        ):
            raise ValueError("final PDF header does not identify the exact retained event and year")
        parsed = parse_final_text(inspected["text"], event, drivers, teams)
        if {row["driver_id"]: row["constructor_id"] for row in parsed.to_pylist()} != constructors:
            raise ValueError("final FIA driver/constructor roster differs from retained qualifying")
        recheck_artifact, recheck_response = fetch(registry_url)
        rechecked = _final(
            _registry(recheck_response.text, event.season, race_name), event, race_name
        )
        if rechecked != selected:
            raise ValueError("final registry version changed while the PDF was captured")
        observed = _clock(now)
        published_bound = _publication(selected) + timedelta(minutes=1)
        if not race_start < published_bound <= observed:
            raise ValueError("final publication bound is before the race or not yet known")
        availability = max(observed, published_bound)
        audit_reference = (
            f"fia-prospective-final-v1:{manifest['manifest_sha256']}:"
            f"{registry_artifact['sha256']}:{final_artifact['sha256']}:{recheck_artifact['sha256']}"
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
                    "label_available_at": availability,
                    "audit_reference": audit_reference,
                }
                for row in parsed.to_pylist()
            ],
            schema=OUTCOME_SCHEMA,
        )
        validate_audited_outcomes(labels, field_roster={(event, driver) for driver in constructors})
        if sum(labels["winner"].to_pylist()) != 1:
            raise ValueError("final FIA classification must contain exactly one winner")
        buffer = pa.BufferOutputStream()
        pq.write_table(labels, buffer)
        data = buffer.getvalue().to_pybytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = (
            Path("data/raw/prospective_scheduler/automatic_outcomes")
            / event.partition()
            / f"{digest}.parquet"
        )
        _immutable(root / relative, data)
        audit = {
            "version": 1,
            "audit_method": "fia-prospective-final-v1",
            "event_id": event.partition(),
            "capture_manifest_sha256": manifest["manifest_sha256"],
            "registry_url": registry_url,
            "document_url": selected["url"],
            "document_id": selected["document_id"],
            "pdf_extraction": {
                "version": inspected["extraction_version"],
                "pages": inspected["pages"],
                "identity_source": "cover-and-layout-text",
            },
            "status": "final",
            "latest_final_audited": True,
            "version_audited": True,
            "final_registry_record": selected,
            "source_observations": observations,
            "printed_registry_clock": selected["publication_cet"],
            "printed_registry_timezone": "CET",
            "conservative_publication_bound_utc": published_bound.isoformat(),
            "label_available_at_utc": availability.isoformat(),
            "audit_reference": audit_reference,
            "exact_driver_aliases": drivers,
            "exact_constructor_aliases": teams,
            "outcomes": {"path": relative.as_posix(), "sha256": digest},
        }
        proof_bytes = _json(audit)
        proof_hash = hashlib.sha256(proof_bytes).hexdigest()
        proof = relative.with_name(f"audit-{proof_hash}.json")
        _immutable(root / proof, proof_bytes)
        return {
            "path": relative.as_posix(),
            "sha256": digest,
            "audit": {"path": proof.as_posix(), "sha256": proof_hash},
            "event_id": event.partition(),
            "document_id": selected["document_id"],
            "label_available_at_utc": availability.isoformat(),
        }
    finally:
        if own:
            client.close()
