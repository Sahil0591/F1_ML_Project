"""Automatic bounded discovery of candidate FIA publication records.

Current Jolpica schedules supply discovery identities and completion hints only.
No retained current response certifies historical availability or Gold eligibility.
"""

import hashlib
import re
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import httpx

from f1_ml_predictor.sources.jolpica import BASE_URL, PAGE_LIMIT
from f1_ml_predictor.trust.historical import (
    CORE_SCHEMA_VERSION,
    _immutable,
    _json,
    _official,
    _retain_response,
)
from f1_ml_predictor.trust.winter import registry_rows

FIA_ROOT = "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14"


class _Selectors(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.options: list[tuple[str, str]] = []
        self.select = False
        self.value: str | None = None
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "select":
            self.select = True
        elif tag == "option" and self.select:
            self.value = dict(attrs).get("value")
            self.text = []

    def handle_data(self, data: str) -> None:
        if self.value is not None:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "option" and self.value is not None:
            self.options.append((self.value, " ".join("".join(self.text).split())))
            self.value = None
        elif tag == "select":
            self.select = False


def _name(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()


def _selector(html: str, label: str, *, season: int, event: bool = False) -> str:
    parser = _Selectors()
    parser.feed(html)
    matches = [value for value, text in parser.options if _name(text) == _name(label)]
    if len(matches) != 1:
        raise ValueError("exact FIA selector is missing or ambiguous")
    url = urljoin("https://www.fia.com", matches[0])
    _official(url)
    path = urlsplit(url).path
    required = rf"/season/season-{season}-\d+"
    if not re.search(required + (r"/event/" if event else r"$"), path):
        raise ValueError("FIA selector does not bind the requested season")
    if not path.startswith(urlsplit(FIA_ROOT).path + "/"):
        raise ValueError("FIA selector is outside the Formula One championship")
    return url


def _jolpica_artifact(root: Path, response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    if response.url.scheme != "https" or response.url.host != "api.jolpi.ca":
        raise ValueError("candidate schedule response is outside official Jolpica API")
    if len(response.content) > 2 * 1024 * 1024:
        raise ValueError("candidate schedule exceeds bounded response size")
    digest = hashlib.sha256(response.content).hexdigest()
    path = root / "data/raw/candidate_discovery/objects" / f"{digest}.json"
    _immutable(path, response.content)
    record = {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "url": str(response.url),
        "provider": "Jolpica",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "request_parameters": dict(response.request.url.params),
        "provider_version": response.headers.get("etag"),
        "last_modified": response.headers.get("last-modified"),
        "evidence_classification": "current_state_only",
    }
    _immutable(
        root
        / "data/raw/candidate_discovery/requests"
        / f"{hashlib.sha256(_json(record)).hexdigest()}.json",
        _json(record),
    )
    return record


def _schedule_rows(response: httpx.Response, season: int) -> list[dict[str, Any]]:
    metadata = response.json()["MRData"]
    total, offset, page_limit = (int(metadata[key]) for key in ("total", "offset", "limit"))
    rows = metadata["RaceTable"]["Races"]
    if (
        offset != 0
        or not 0 <= total <= PAGE_LIMIT
        or page_limit != PAGE_LIMIT
        or not isinstance(rows, list)
        or len(rows) != total
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("season schedule pagination is incomplete or exceeds discovery bound")
    if metadata["RaceTable"].get("season") != str(season):
        raise ValueError("season schedule response contradicts the requested season")
    return rows


def _hint(row: dict[str, Any], season: int, now: datetime) -> dict[str, Any]:
    if row.get("season") != str(season):
        raise ValueError("race identity contradicts schedule season")
    round_number = int(row["round"])
    circuit = row["Circuit"]["circuitId"]
    event_name = row["raceName"]
    if (
        not 1 <= round_number <= 30
        or not isinstance(event_name, str)
        or not event_name.strip()
        or not isinstance(circuit, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]*", circuit)
    ):
        raise ValueError("invalid discovery race identity")
    race_start = datetime.fromisoformat(f"{row['date']}T{row['time']}".replace("Z", "+00:00"))
    if race_start.tzinfo is None:
        raise ValueError("schedule completion hint has no timezone")
    race_start = race_start.astimezone(UTC)
    if race_start >= now:
        raise ValueError("race_not_completed_as_of_discovery")
    if race_start.year != season:
        raise ValueError("schedule_date_year_contradicts_event_identity")
    # Used for ranking only. No historical session clock is certified here.
    q_date = row.get("Qualifying", {}).get("date", row["date"])
    q_day = datetime.fromisoformat(q_date)
    winter_hint = q_day.replace(tzinfo=ZoneInfo("Europe/Paris")).utcoffset() == timedelta(hours=1)
    return {
        "season": season,
        "round": round_number,
        "circuit_id": circuit,
        "event_name": event_name,
        "discovery_race_start_hint_utc": race_start.isoformat(),
        "discovery_qualifying_date_hint": q_date,
        "winter_ranking_hint": winter_hint,
        "identity_evidence_classification": "current_state_only",
        "identity_requires_official_schedule_audit": True,
    }


def _stem(event_name: str, season: int) -> str:
    ascii_name = unicodedata.normalize("NFKD", event_name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return f"/system/files/decision-document/{season}_{slug}_-_"


def _clock(row: dict[str, Any]) -> tuple[datetime, bool]:
    local = datetime.strptime(row["publication_cet"], "%d.%m.%y %H:%M")
    unambiguous = local.replace(tzinfo=ZoneInfo("Europe/Paris")).utcoffset() == timedelta(hours=1)
    return local.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC), unambiguous


def _inspect_registry(candidate: dict[str, Any], html: str) -> None:
    prefix = _stem(candidate["event_name"], candidate["season"])
    rows = registry_rows(html)
    matching = [
        row for row in rows if row.get("url") and urlsplit(row["url"]).path.startswith(prefix)
    ]
    for row in matching:
        _official(row["url"])
    if not matching:
        raise ValueError("registry_document_identity_does_not_match_exact_event_and_season")
    qualification = [
        row
        for row in matching
        if "qualifying classification" in row["title"].lower()
        and "sprint" not in row["title"].lower()
    ]
    # Recalled rows have no URL. Preserve them separately without guessing their event.
    candidate["unbound_recalled_records"] = [
        row for row in rows if row["recalled"] and not row["url"]
    ]
    candidate["registry_documents"] = matching
    provisionals = [
        row
        for row in qualification
        if "provisional" in row["title"].lower() and not row["recalled"]
    ]
    if not provisionals:
        raise ValueError("downloadable_provisional_qualifying_record_missing")
    selected = min(provisionals, key=lambda row: (_clock(row)[0], int(row["document_id"])))
    published, winter = _clock(selected)
    if published.year != candidate["season"]:
        raise ValueError("qualifying_publication_year_contradicts_event_identity")
    candidate.update(
        {
            "qualifying_url": selected["url"],
            "document_id": selected["document_id"],
            "publication_cet": selected["publication_cet"],
            "publication_precision_seconds": 60,
            "publication_timezone_status": "verified_literal_CET"
            if winter
            else "unresolved_CET_CEST",
            "published_at_utc": published.isoformat() if winter else None,
            "prediction_timestamp_utc": (published + timedelta(minutes=2)).isoformat()
            if winter
            else None,
            "winter_publication_record": winter,
            "qualifying_versions": qualification,
            "entry_list_records": [row for row in matching if "entry list" in row["title"].lower()],
            "final_race_records": [
                row
                for row in matching
                if "final race classification" in row["title"].lower()
                and "sprint" not in row["title"].lower()
                and not row["recalled"]
            ],
        }
    )
    reasons = [
        "current_schedule_identity_requires_official_audit",
        "exact_pdf_values_not_inspected",
        "latest_required_field_state_not_audited",
        "roster_completeness_not_inspected",
        "final_labels_not_audited",
        "schedule_publication_not_audited",
    ]
    if not winter:
        reasons.append("summer_publication_clock_unresolved")
    if not candidate["final_race_records"]:
        reasons.append("final_race_classification_missing")
    if candidate["unbound_recalled_records"]:
        reasons.append("recalled_record_event_binding_requires_review")
    near_versions = [
        row
        for row in qualification
        if row != selected and _clock(row)[0] <= published + timedelta(minutes=2)
    ]
    if near_versions:
        reasons.append("another_qualifying_version_at_proposed_cutoff")
    candidate["exclusion_reasons"] = reasons
    candidate["inclusion_reasons"] = [
        "completed_race_discovery_hint",
        "exact_official_event_selector",
        "downloadable_provisional_qualifying_record",
    ]
    if winter:
        candidate["inclusion_reasons"].append("winter_publication_clock_unambiguous")
    candidate["discovery_score"] = (
        (100 if winter else 0)
        + (20 if candidate["final_race_records"] else 0)
        + (5 if candidate["entry_list_records"] else 0)
        - (20 if near_versions else 0)
    )


def discover_candidates(
    root: Path,
    seasons: tuple[int, ...] = (2026, 2025),
    limit: int = 17,
    http_client: httpx.Client | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Retain schedules and at most twenty event registries, without PDF downloads."""
    if isinstance(limit, bool) or not 1 <= limit <= 20:
        raise ValueError("candidate discovery limit must be between one and twenty")
    if (
        not seasons
        or len(seasons) > 3
        or len(set(seasons)) != len(seasons)
        or any(
            isinstance(season, bool) or not isinstance(season, int) or not 2015 <= season <= 2100
            for season in seasons
        )
    ):
        raise ValueError("discover one to three distinct supported seasons")
    as_of = now or datetime.now(UTC)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("candidate discovery time requires a timezone")
    as_of = as_of.astimezone(UTC)
    root = root.resolve()
    own = http_client is None
    client = http_client or httpx.Client(timeout=30, follow_redirects=False)
    artifacts: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    hints: list[dict[str, Any]] = []
    cache: dict[str, str] = {}
    failed_urls: dict[str, str] = {}

    def fia_html(url: str) -> str:
        _official(url)
        if url in failed_urls:
            raise ValueError(failed_urls[url])
        if url not in cache:
            if own:
                time.sleep(0.25)
            try:
                response = client.get(url, headers={"User-Agent": "f1-ml-predictor/0.1.0"})
                _official(str(response.url))
                artifact = _retain_response(root, response)
                if artifact["provider"] != "FIA" or response.content.startswith(b"%PDF"):
                    raise ValueError("candidate registry response is not official FIA HTML")
            except (ValueError, httpx.HTTPError) as exc:
                failed_urls[url] = str(exc)
                raise
            artifacts.append(artifact)
            cache[url] = response.text
        return cache[url]

    try:
        for season in seasons:
            try:
                response = client.get(
                    f"{BASE_URL}/{season}/",
                    params={"limit": PAGE_LIMIT, "offset": 0},
                    headers={"User-Agent": "f1-ml-predictor/0.1.0"},
                )
                artifacts.append(_jolpica_artifact(root, response))
                rows = _schedule_rows(response, season)
                round_counts = Counter(str(row.get("round")) for row in rows)
                for row in rows:
                    try:
                        hint = _hint(row, season, as_of)
                        if round_counts[str(row.get("round"))] != 1:
                            raise ValueError("duplicate_schedule_round")
                        hints.append(hint)
                    except (ValueError, KeyError, TypeError) as exc:
                        excluded.append({"season": season, "schedule_row": row, "reason": str(exc)})
            except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                errors.append({"season": season, "reason": str(exc)})
        hints.sort(
            key=lambda item: (not item["winter_ranking_hint"], -item["season"], -item["round"])
        )
        pool = hints[:limit]
        excluded.extend({**item, "reason": "bounded_pool_limit"} for item in hints[limit:])
        try:
            root_html = fia_html(FIA_ROOT)
        except (ValueError, httpx.HTTPError) as exc:
            root_html = ""
            errors.append({"url": FIA_ROOT, "reason": str(exc)})
        season_pages: dict[int, str] = {}
        for candidate in pool:
            candidate.update(
                {
                    "status": "audit_required",
                    "eligible_gold": False,
                    "discovery_score": 0,
                    "exclusion_reasons": [],
                }
            )
            try:
                season = candidate["season"]
                if season not in season_pages:
                    season_url = _selector(root_html, f"SEASON {season}", season=season)
                    season_pages[season] = fia_html(season_url)
                index_url = _selector(
                    season_pages[season], candidate["event_name"], season=season, event=True
                )
                candidate["index_url"] = index_url
                _inspect_registry(candidate, fia_html(index_url))
            except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                candidate["status"] = "excluded"
                candidate["exclusion_reasons"] = [str(exc)]
        pool.sort(key=lambda item: (-item["discovery_score"], -item["season"], -item["round"]))
    finally:
        if own:
            client.close()
    catalog = {
        "version": 1,
        "discovery_method": "automatic-fia-registry-v1",
        "as_of_utc": as_of.isoformat(),
        "seasons": list(seasons),
        "pool_limit": limit,
        "candidates": pool,
        "excluded_events": excluded,
        "source_errors": errors,
        "source_artifacts": artifacts,
        "eligible_gold_races": 0,
        "evidence_classification": "current_state_only",
    }
    catalog_hash = hashlib.sha256(_json(catalog)).hexdigest()
    catalog_path = root / "data/benchmarks/historical_audit" / f"candidates-{catalog_hash}.json"
    _immutable(catalog_path, _json(catalog))
    report = {
        "version": 1,
        "schema_version": CORE_SCHEMA_VERSION,
        "minimum_gold_races": 8,
        "eligible_gold_races": 0,
        "status": "audit_required"
        if any(item["status"] == "audit_required" for item in pool)
        else "insufficient_data",
        "candidate_count": len(pool),
        "preferred_pool_size_met": 12 <= len(pool) <= 20,
        "candidates_with_publication_records": sum(
            item["status"] == "audit_required" for item in pool
        ),
        "winter_publication_candidates": sum(
            item.get("winter_publication_record", False) for item in pool
        ),
        "catalog_path": catalog_path.relative_to(root).as_posix(),
        "catalog_sha256": catalog_hash,
        "candidates": pool,
        "source_errors": errors,
    }
    report_hash = hashlib.sha256(_json(report)).hexdigest()
    report_path = (
        root / "data/benchmarks/historical_audit" / f"candidate-auditability-{report_hash}.json"
    )
    _immutable(report_path, _json(report))
    return {
        "catalog": catalog,
        "catalog_path": catalog_path.relative_to(root).as_posix(),
        "catalog_sha256": catalog_hash,
        "auditability_report": report,
        "report_path": report_path.relative_to(root).as_posix(),
        "report_sha256": report_hash,
    }
