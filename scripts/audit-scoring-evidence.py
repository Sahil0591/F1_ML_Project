"""Collect and audit official championship scoring evidence for 2022-2026.

``collect`` retains FIA event registries, every FIA "Championship Points" PDF
(including after-sprint, revised and ICA-revised versions), stewards' review
documents, FIA sporting regulation issues, ICA judgements, formula1.com race,
sprint and standings pages, and Jolpica identity crosswalk responses.

``build`` is offline and deterministic. It parses only retained bytes and
writes ``data/audit/event_points_evidence.json`` and
``data/audit/scoring_rules.json``. FIA documents are the scoring authority;
formula1.com is secondary confirmation; Jolpica is used for identity checks
and tertiary comparison only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin

import httpx
from pypdf import PdfReader

from f1_ml_predictor.trust.candidates import FIA_ROOT, _Selectors
from f1_ml_predictor.trust.historical import _immutable, _json, _retain_response
from f1_ml_predictor.trust.winter import (
    DRIVER_ALIASES,
    constructor_aliases_for_season,
    registry_rows,
)

SEASONS = (2022, 2023, 2024, 2025, 2026)
FIA_SEASON_PATHS = {
    2022: "season-2022-2005",
    2023: "season-2023-2042",
    2024: "season-2024-2043",
    2025: "season-2025-2071",
    2026: "season-2026-2072",
}
# The audit is frozen at this instant; later events are out of scope.
AUDIT_AS_OF = datetime(2026, 10, 7, tzinfo=UTC)
JOLPICA = "https://api.jolpi.ca/ergast/f1"
F1_RESULTS = "https://www.formula1.com/en/results"
FIA_REGULATIONS = "https://www.fia.com/regulation/category/110"
FIA_ICA_JUDGEMENTS = "https://www.fia.com/judgements-ica"
HEADERS = {"User-Agent": "Mozilla/5.0 (f1-ml-predictor scoring evidence audit)"}
POINTS_TITLE = re.compile(r"championship points", re.I)
REVIEW_TITLE = re.compile(
    r"right of review|\bICA\b|court of appeal|appeal|revised|reinstat|post-race checks",
    re.I,
)
PARSER_VERSION = "fia-championship-points-matrix-v2"
SCHEMA_VERSION = "scoring-evidence-v1"
EXTRA_FIA_EVENT_NAMES = {
    # Jolpica names that differ from FIA registry selector labels.
    (2022, "Mexico City Grand Prix"): "Mexican Grand Prix",
    (2022, "São Paulo Grand Prix"): "Brazilian Grand Prix",
    (2026, "Barcelona Grand Prix"): "Barcelona-Catalunya Grand Prix",
}
F1_SLUGS = {
    "Bahrain Grand Prix": "bahrain",
    "Saudi Arabian Grand Prix": "saudi-arabia",
    "Australian Grand Prix": "australia",
    "Emilia Romagna Grand Prix": "emilia-romagna",
    "Miami Grand Prix": "miami",
    "Spanish Grand Prix": "spain",
    "Monaco Grand Prix": "monaco",
    "Azerbaijan Grand Prix": "azerbaijan",
    "Canadian Grand Prix": "canada",
    "British Grand Prix": "great-britain",
    "Austrian Grand Prix": "austria",
    "French Grand Prix": "france",
    "Hungarian Grand Prix": "hungary",
    "Belgian Grand Prix": "belgium",
    "Dutch Grand Prix": "netherlands",
    "Italian Grand Prix": "italy",
    "Singapore Grand Prix": "singapore",
    "Japanese Grand Prix": "japan",
    "United States Grand Prix": "united-states",
    "Mexico City Grand Prix": "mexico",
    "São Paulo Grand Prix": "brazil",
    "Abu Dhabi Grand Prix": "abu-dhabi",
    "Qatar Grand Prix": "qatar",
    "Las Vegas Grand Prix": "las-vegas",
    "Chinese Grand Prix": "china",
    "Barcelona Grand Prix": "barcelona-catalunya",
}
# FIA matrix header codes used for each event across 2022-2026.
EVENT_CODES = {
    "Bahrain Grand Prix": {"BHR", "BRN"},
    "Saudi Arabian Grand Prix": {"SAU", "KSA"},
    "Australian Grand Prix": {"AUS"},
    "Emilia Romagna Grand Prix": {"ITA", "EMI"},
    "Miami Grand Prix": {"USA", "MIA"},
    "Spanish Grand Prix": {"ESP"},
    "Monaco Grand Prix": {"MCO", "MON"},
    "Azerbaijan Grand Prix": {"AZE"},
    "Canadian Grand Prix": {"CAN"},
    "British Grand Prix": {"GBR"},
    "Austrian Grand Prix": {"AUT"},
    "French Grand Prix": {"FRA"},
    "Hungarian Grand Prix": {"HUN"},
    "Belgian Grand Prix": {"BEL"},
    "Dutch Grand Prix": {"NLD", "NED"},
    "Italian Grand Prix": {"ITA"},
    "Singapore Grand Prix": {"SGP", "SIN"},
    "Japanese Grand Prix": {"JPN"},
    "United States Grand Prix": {"USA"},
    "Mexico City Grand Prix": {"MEX"},
    "São Paulo Grand Prix": {"BRA"},
    "Brazilian Grand Prix": {"BRA"},
    "Abu Dhabi Grand Prix": {"UAE", "ABU"},
    "Qatar Grand Prix": {"QAT"},
    "Las Vegas Grand Prix": {"USA", "LVG"},
    "Chinese Grand Prix": {"CHN"},
    "Barcelona Grand Prix": {"ESP"},
    "Bahrain Grand Prix in Malaysia": {"BHR", "MAS", "MYS"},
}
JOLPICA_CONSTRUCTORS = {
    "alfa": "alfa_romeo",
    "alphatauri": "alpha_tauri",
}


def _norm(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(folded.replace("-", " ").split()).casefold()


_RELOCATED_EVENT = re.compile(r"^(?P<base>.+? Grand Prix) in \S.*$")


def _name_variants(name: str) -> list[str]:
    """The Jolpica name, then the original event name for a relocated event.

    A relocated event such as "Bahrain Grand Prix in Malaysia" keeps its original
    name on the FIA registry and on formula1.com.
    """
    match = _RELOCATED_EVENT.match(name)
    return [name, match["base"]] if match else [name]


def _event_codes(name: str) -> set[str]:
    return set().union(*(EVENT_CODES.get(variant, set()) for variant in _name_variants(name)))


def _surname_key(value: str) -> str:
    return _norm(value).upper()


# Surnames identify drivers uniquely within 2022-2026; the Gold alias table is
# the single source of driver identifiers.
SURNAMES: dict[str, str] = {}
for _alias, _driver in DRIVER_ALIASES.items():
    _words = _alias.split()
    _upper = [word for word in _words if word == word.upper() and len(word) > 1]
    for _key in {" ".join(_upper), _words[-1]}:
        if _key:
            SURNAMES[_surname_key(_key)] = _driver


def driver_from_name(name: str) -> str | None:
    """Map 'M. VERSTAPPEN', 'N. DE VRIES' or a surname onto a Gold driver id."""
    words = [word for word in name.replace(".", ". ").split() if not re.fullmatch(r"[A-Z]\.", word)]
    for start in range(len(words)):
        key = _surname_key(" ".join(words[start:]))
        if key in SURNAMES:
            return SURNAMES[key]
    return None


# ---------------------------------------------------------------- collection


def _retain_jolpica(root: Path, response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    path = root / "data/raw/scoring_audit/jolpica/objects" / f"{digest}.json"
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
        "content_type": response.headers.get("content-type"),
        "evidence_classification": "current_state_only",
    }
    _immutable(
        root
        / "data/raw/scoring_audit/jolpica/requests"
        / f"{hashlib.sha256(_json(record)).hexdigest()}.json",
        _json(record),
    )
    return record


def _get(client: httpx.Client, url: str, *, pause: float = 0.0) -> httpx.Response:
    for attempt in range(5):
        response = client.get(url)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if pause:
            time.sleep(pause)
        return response
    response.raise_for_status()
    return response


def _pdf_links(html: str, base: str) -> list[tuple[str, str]]:
    links = []
    for href, text in re.findall(r'href="([^"]+\.pdf)"[^>]*>(.*?)</a>', html, re.S):
        title = " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())
        links.append((urljoin(base, unescape(href)), title))
    return list(dict.fromkeys(links))


def _collect_regulations(root: Path, client: httpx.Client) -> dict[str, Any]:
    listing = _get(client, FIA_REGULATIONS)
    record = {"listing_page": _retain_response(root, listing), "documents": []}
    seen = set()
    for url, title in _pdf_links(listing.text, "https://www.fia.com"):
        season = re.search(r"\b(2022|2023|2024|2025|2026)\b", title)
        lowered = (url + " " + title).lower()
        if not season or url in seen or " pu " in f" {title.lower()} ":
            continue
        if "2027" in title:
            continue
        sporting = "sporting" in lowered
        general = season[1] == "2026" and "section_a" in lowered
        if not (sporting or general) or (
            season[1] == "2026" and "section_b" not in lowered and not general
        ):
            continue
        seen.add(url)
        document = _retain_response(root, _get(client, url))
        document["listing_title"] = title
        record["documents"].append(document)
    return record


def _collect_ica(root: Path, client: httpx.Client) -> dict[str, Any]:
    listing = _get(client, FIA_ICA_JUDGEMENTS)
    record = {"listing_page": _retain_response(root, listing), "documents": []}
    for url, _title in _pdf_links(listing.text, "https://www.fia.com"):
        record["documents"].append(_retain_response(root, _get(client, url)))
    return record


def collect(root: Path) -> Path:
    root = root.resolve()
    client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=90)
    index: dict[str, Any] = {
        "audit_as_of_utc": AUDIT_AS_OF.isoformat(),
        "seasons": {},
        "regulations": _collect_regulations(root, client),
        "ica_judgements": _collect_ica(root, client),
    }
    for season in SEASONS:
        schedule_response = _get(client, f"{JOLPICA}/{season}.json?limit=100", pause=0.5)
        schedule_record = _retain_jolpica(root, schedule_response)
        races = schedule_response.json()["MRData"]["RaceTable"]["Races"]
        season_url = f"{FIA_ROOT}/season/{FIA_SEASON_PATHS[season]}"
        season_response = _get(client, season_url)
        season_record = _retain_response(root, season_response)
        selectors = _Selectors()
        selectors.feed(season_response.text)
        fia_events = {
            text: urljoin("https://www.fia.com", value)
            for value, text in selectors.options
            if "/event/" in value and FIA_SEASON_PATHS[season] in value
        }
        f1_races_response = _get(client, f"{F1_RESULTS}/{season}/races")
        f1_races_record = _retain_response(root, f1_races_response)
        f1_links = re.findall(
            rf"/en/results/{season}/races/\d+/([a-z0-9-]+)/race-result", f1_races_response.text
        )
        f1_paths = {
            slug: path
            for path, slug in zip(
                re.findall(
                    rf"/en/results/{season}/races/\d+/[a-z0-9-]+/race-result",
                    f1_races_response.text,
                ),
                f1_links,
                strict=True,
            )
        }
        standings = {
            kind: _retain_response(root, _get(client, f"{F1_RESULTS}/{season}/{kind}"))
            for kind in ("drivers", "team")
        }
        events = []
        completed = [
            race
            for race in races
            if datetime.fromisoformat(race["date"]).replace(tzinfo=UTC) < AUDIT_AS_OF
        ]
        upcoming = [race for race in races if race not in completed][:1]
        for race in completed + upcoming:
            round_number = int(race["round"])
            name = race["raceName"]
            candidates = list(
                dict.fromkeys(
                    [EXTRA_FIA_EVENT_NAMES.get((season, name), name), *_name_variants(name)]
                )
            )
            fia_name, matches = candidates[0], []
            for candidate in candidates:
                found = [url for text, url in fia_events.items() if _norm(text) == _norm(candidate)]
                if len(found) == 1:
                    fia_name, matches = candidate, found
                    break
            event: dict[str, Any] = {
                "season": season,
                "round": round_number,
                "event_id": f"season={season}/round={round_number:02d}",
                "event_name": name,
                "fia_event_name": fia_name,
                "race_date": race["date"],
                "has_sprint": "Sprint" in race,
                "completed": race in completed,
            }
            if len(matches) != 1:
                event["fia_event_page"] = None
                event["collection_error"] = "fia_event_selector_missing_or_ambiguous"
                events.append(event)
                continue
            page_response = _get(client, matches[0])
            event["fia_event_page"] = _retain_response(root, page_response)
            rows = registry_rows(page_response.text)
            documents = []
            for row in rows:
                if not (
                    POINTS_TITLE.search(row["title"])
                    or POINTS_TITLE.search(unquote(row.get("url") or "").replace("_", " "))
                    or (
                        REVIEW_TITLE.search(row["title"])
                        and "entry list" not in row["title"].lower()
                    )
                ):
                    continue
                entry = dict(row)
                entry["kind"] = "points" if POINTS_TITLE.search(row["title"]) else "review"
                if row["url"] and not row["recalled"]:
                    entry["retained"] = _retain_response(root, _get(client, row["url"]))
                documents.append(entry)
            event["documents"] = documents
            event["classification_documents"] = [
                row
                for row in rows
                if re.search(r"classification", row["title"], re.I)
                and re.search(r"race|sprint", row["title"], re.I)
                and not re.search(r"qualifying|shootout", row["title"], re.I)
            ]
            if not event["completed"]:
                events.append(event)
                continue
            for kind in ("results", "sprint"):
                if kind == "sprint" and not event["has_sprint"]:
                    continue
                url = f"{JOLPICA}/{season}/{round_number}/{kind}.json?limit=100"
                event[f"jolpica_{kind}"] = _retain_jolpica(root, _get(client, url, pause=0.5))
            slug = next(
                (F1_SLUGS[variant] for variant in _name_variants(name) if variant in F1_SLUGS),
                None,
            )
            if slug in f1_paths:
                race_url = f"https://www.formula1.com{f1_paths[slug]}"
                pages: dict[str, Any] = {
                    "race_result_url": race_url,
                    "race": _retain_response(root, _get(client, race_url)),
                }
                if event["has_sprint"]:
                    sprint_url = race_url.removesuffix("race-result") + "sprint-results"
                    pages["sprint"] = _retain_response(root, _get(client, sprint_url))
                event["formula1"] = pages
            else:
                event["formula1"] = None
            events.append(event)
        index["seasons"][str(season)] = {
            "jolpica_schedule": schedule_record,
            "fia_season_page": season_record,
            "formula1_races_page": f1_races_record,
            "formula1_standings": standings,
            "events": events,
        }
    payload = _json(index)
    path = (
        root / "data/raw/scoring_audit" / f"collection-{hashlib.sha256(payload).hexdigest()}.json"
    )
    _immutable(path, payload)
    return path


# ------------------------------------------------------------------ parsing


def _runs(page: Any) -> list[tuple[float, float, str]]:
    runs: list[tuple[float, float, str]] = []

    def visit(text: str, cm: list[float], tm: list[float], _font: Any, _size: Any) -> None:
        if not text.strip():
            return
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        runs.append((round(x, 2), round(y, 2), text.strip()))

    page.extract_text(visitor_text=visit)
    # Landscape matrices are stored as rotated pages. Normalize them so that
    # x grows rightward along event columns and y decreases down the rows.
    rotation = page.rotation % 360
    if rotation == 90:
        runs = [(y, -x, text) for x, y, text in runs]
    elif rotation == 270:
        runs = [(-y, x, text) for x, y, text in runs]
    elif rotation != 0:
        raise ValueError("unsupported points matrix page rotation")
    return runs


_POSITION = re.compile(r"^(?:F\s*)?(\d{1,2}|NC|DQ|DSQ|EX|DNS|DNQ|DNF|WD|NS|DNP)(?:\s*F)?$")
_INTEGER = re.compile(r"^-?\d+(?:\.\d+)?$")


def _grid(xs: list[float]) -> tuple[float, float]:
    """Fit a regular column grid; the leftmost cluster is the first event."""
    clusters: list[list[float]] = []
    for x in sorted(xs):
        if clusters and x - clusters[-1][-1] <= 9:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    centers = [statistics.median(cluster) for cluster in clusters]
    if len(centers) == 1:
        return centers[0], 0.0
    gaps = [b - a for a, b in zip(centers, centers[1:], strict=False)]
    unit = min(gaps)
    steps = [round(gap / unit) for gap in gaps]
    pitch = (centers[-1] - centers[0]) / sum(steps)
    if any(abs(gap - step * pitch) > 6 for gap, step in zip(gaps, steps, strict=True)):
        raise ValueError("points matrix columns are not on a regular grid")
    return centers[0], pitch


def _column(x: float, origin: float, pitch: float) -> int:
    if pitch == 0:
        if abs(x - origin) > 9:
            raise ValueError("single-column matrix cell is outside the column")
        return 0
    index = (x - origin) / pitch
    if abs(index - round(index)) > 0.3:
        raise ValueError("matrix cell is ambiguous between columns")
    return round(index)


def _cover(reader: PdfReader) -> dict[str, Any]:
    text = reader.pages[0].extract_text() or ""
    flat = " ".join(text.split())
    document = re.search(r"Document\s+(\d+)", flat) or re.search(r"Doc\s+(\d+)\s+Time", flat)
    date = re.search(r"Date\s+(\d{1,2} \w+ \d{4})", flat)
    clock = re.search(r"Time\s+(\d{1,2}:\d{2})", flat)
    title = re.search(r"Title\s+(.*?)\s+Description", flat)
    return {
        "cover_document_number": document[1] if document else None,
        "cover_local_date": date[1] if date else None,
        "cover_local_time": clock[1] if clock else None,
        "cover_title": title[1] if title else None,
    }


def parse_points_pdf(content: bytes) -> dict[str, Any]:
    reader = PdfReader(BytesIO(content))
    result: dict[str, Any] = {"drivers": [], "entrants": [], "headers": None, "notes": []}
    result.update(_cover(reader))
    blocks: dict[str, list[dict[str, Any]]] = {"drivers": [], "entrants": []}
    for page in reader.pages:
        runs = _runs(page)
        starts = [run for run in runs if re.match(r"^(DRIVER|ENTRANT)\b", run[2])]
        if not starts:
            continue
        # Some pages draw the matrix rotated inside the content stream while
        # declaring no /Rotate; driver names then share the header's line.
        if any(
            abs(run[1] - starts[0][1]) <= 0.6
            and run[0] > starts[0][0] + 5
            and re.search(r"[a-z.]", run[2])
            for run in runs
        ):
            runs = [(y, -x, text) for x, y, text in runs]
            starts = [run for run in runs if re.match(r"^(DRIVER|ENTRANT)\b", run[2])]
        # Some generators emit the header as one run, others as one run per word.
        hx, hy, _ = starts[0]
        line = sorted(
            (run for run in runs if abs(run[1] - hy) <= 0.6 and run[0] >= hx),
            key=lambda run: run[0],
        )
        htext = " ".join(" ".join(run[2] for run in line).split())
        if not re.match(r"^(DRIVER|ENTRANT)\s+TOTAL\b", htext):
            continue
        kind = "drivers" if htext.startswith("DRIVER") else "entrants"
        codes = htext.split()[2:]
        if result["headers"] is None:
            result["headers"] = codes
        elif result["headers"] != codes:
            raise ValueError("points matrix pages disagree on event headers")
        # The legal footer ("The F1 FORMULA 1 logo ... trade marks") splits into
        # numeric fragments; everything at or below its top line is excluded.
        footer = [
            run[1]
            for run in runs
            if re.search(
                r"trade mark|All rights reserved|World Championship Limited"
                r"|Page \d+ of \d+|The Stewards",
                run[2],
            )
            and run[1] < hy
        ]
        floor = max(footer) + 1 if footer else float("-inf")
        body = [run for run in runs if floor < run[1] < hy - 2]
        for run in body:
            text = " ".join(run[2].split())
            if (
                re.search(r"subject to|appeal|under review", text, re.I)
                and text not in result["notes"]
            ):
                result["notes"].append(text)

        def is_anchor(
            run: tuple[float, float, str],
            body: list[tuple[float, float, str]] = body,
            hx: float = hx,
        ) -> bool:
            # A rank is followed on its line by a name and then a numeric total.
            # This rejects the footer, whose "F1" logo yields a lone "1".
            if not (run[0] < hx - 2 and re.fullmatch(r"\d{1,2}", run[2])):
                return False
            line = sorted(
                (r for r in body if abs(r[1] - run[1]) <= 0.6 and r[0] > run[0]), key=lambda r: r[0]
            )
            first_number = next((r for r in line if _INTEGER.match(r[2])), None)
            if first_number is None:
                return False
            # Entrant names wrap onto the lines above and below the rank.
            names = [
                r
                for r in body
                if abs(r[1] - run[1]) <= 8
                and run[0] < r[0] < first_number[0]
                and re.search(r"[A-Za-z]", r[2])
            ]
            return bool(names)

        anchors = sorted((run for run in body if is_anchor(run)), key=lambda run: -run[1])
        if not anchors:
            continue
        totals = {}
        for ax, ay, _rank in anchors:
            row = [run for run in body if abs(run[1] - ay) <= 0.6 and run[0] > ax]
            numeric = [run for run in row if _INTEGER.match(run[2])]
            if not numeric:
                raise ValueError("matrix row has no printed total")
            totals[(ax, ay)] = min(numeric, key=lambda run: run[0])
        total_x = max(run[0] for run in totals.values())
        cells = [run for run in body if total_x + 12 < run[0] < 800]
        for index, (ax, ay, rank) in enumerate(anchors):
            above = anchors[index - 1][1] if index else None
            below = anchors[index + 1][1] if index + 1 < len(anchors) else None
            upper = (above + ay) / 2 if above is not None else hy - 2
            if below is not None:
                lower = (ay + below) / 2
            else:
                # Position rows sit at most ~10 below the anchor; footers lie further down.
                lower = ay - 12.5
            name_runs = [
                run
                for run in body
                if lower < run[1] < upper
                and ax < run[0] < total_x - 1
                and not _INTEGER.match(run[2])
            ]
            name = " ".join(run[2] for run in sorted(name_runs, key=lambda run: (-run[1], run[0])))
            blocks[kind].append(
                {
                    "rank": int(rank),
                    "name": " ".join(name.split()),
                    "total": float(totals[(ax, ay)][2]),
                    "anchor_y": ay,
                    "cells": [run for run in cells if lower < run[1] < upper],
                }
            )
    if result["headers"] is None:
        if sum(len(_runs(page)) for page in reader.pages[1:]) < 40:
            raise ValueError("image_only_matrix_no_extractable_text")
        raise ValueError("no championship points matrix found")
    codes = result["headers"]
    for kind, kind_blocks in blocks.items():
        if not kind_blocks:
            continue
        # Points sit above each anchor line; entrant position rows can rise to
        # about +3 when a team name wraps onto three lines.
        threshold = 1.0 if kind == "drivers" else 6.0

        def is_points(
            block: dict[str, Any], y: float, text: str, threshold: float = threshold
        ) -> bool:
            return y - block["anchor_y"] > threshold and bool(_INTEGER.match(text))

        position_xs = [
            x
            for block in kind_blocks
            for x, y, text in block["cells"]
            if not is_points(block, y, text) and re.search(r"\d|NC|DQ|EX|DNS", text)
        ]
        origin, pitch = _grid(position_xs)
        if pitch == 0:
            # Only one classified column (e.g. a round-two after-sprint document,
            # whose sprint column prints points without positions): take the
            # pitch from the points cells, whose first cluster is the same column.
            points_xs = [
                x
                for block in kind_blocks
                for x, y, text in block["cells"]
                if is_points(block, y, text)
            ]
            if points_xs:
                _points_origin, pitch = _grid(points_xs)
        offsets = []
        for block in kind_blocks:
            positions = [
                x for x, y, text in block["cells"] if not is_points(block, y, text) and text != "F"
            ]
            for x, y, text in block["cells"]:
                if is_points(block, y, text):
                    right = [p - x for p in positions if 0 < p - x < (pitch or 29) * 0.8]
                    if right:
                        offsets.append(min(right))
        offset = statistics.median(offsets) if offsets else 0.0
        for block in kind_blocks:
            points: dict[int, float] = {}
            positions: dict[int, list[str]] = defaultdict(list)
            for x, y, text in block["cells"]:
                if is_points(block, y, text):
                    column = _column(x + offset, origin, pitch)
                    if column in points:
                        raise ValueError("two points values in one matrix cell")
                    points[column] = float(text)
                elif text == "F":
                    positions[_column(x + (pitch or 29) * 0.3, origin, pitch)].append(text)
                elif _POSITION.match(text):
                    positions[_column(x, origin, pitch)].append(text)
                else:
                    raise ValueError(f"unrecognized position cell {text!r}")
            if any(column >= len(codes) or column < 0 for column in [*points, *positions]):
                raise ValueError("matrix cell lies beyond the event headers")
            computed = sum(points.values())
            result[kind].append(
                {
                    "rank": block["rank"],
                    "name": block["name"],
                    "total": block["total"],
                    "points": {str(k): v for k, v in sorted(points.items())},
                    "positions": {str(k): " ".join(v) for k, v in sorted(positions.items())},
                    "total_matches_sum": abs(computed - block["total"]) < 1e-9,
                    "sum_of_cells": computed,
                }
            )
    return result


def _publication_upper_bound(cet: str) -> datetime:
    """Registry clocks are Paris wall time; UTC+1 is the later bound for CET/CEST."""
    local = datetime.strptime(cet, "%d.%m.%y %H:%M")
    return local.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC)


def _align(codes: list[str], rounds: list[dict[str, Any]]) -> dict[int, int]:
    """Longest common subsequence of header codes and the calendar order."""
    n, m = len(codes), len(rounds)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if codes[i] in _event_codes(rounds[j]["event_name"]):
                table[i][j] = 1 + table[i + 1][j + 1]
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])
    mapping, i, j = {}, 0, 0
    while i < n and j < m:
        if (
            codes[i] in _event_codes(rounds[j]["event_name"])
            and table[i][j] == 1 + table[i + 1][j + 1]
        ):
            mapping[i] = rounds[j]["round"]
            i, j = i + 1, j + 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return mapping


# ----------------------------------------------------------- secondary pages


def _f1_table(path: str) -> list[dict[str, Any]]:
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    rows = []
    for raw in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)[1:]:
        cells = [unescape(cell).strip() for cell in re.sub(r"<[^>]+>", "\t", raw).split("\t")]
        cells = [cell for cell in cells if cell]
        if len(cells) < 6:
            continue
        rows.append(cells)
    return rows


def _f1_result_rows(path: str) -> list[dict[str, Any]]:
    out = []
    for cells in _f1_table(path):
        # Pos | No | First | Last | CODE | Team | Laps | Time | Pts
        driver = driver_from_name(cells[3]) or driver_from_name(cells[2])
        out.append(
            {
                "position_text": cells[0],
                "car_number": cells[1],
                "driver": driver,
                "team": cells[5],
                "points": float(cells[-1]) if _INTEGER.match(cells[-1]) else None,
            }
        )
    return out


def _f1_standings(path: str, kind: str) -> list[dict[str, Any]]:
    out = []
    for cells in _f1_table(path) if kind == "drivers" else []:
        out.append(
            {
                "driver": driver_from_name(cells[2]) or driver_from_name(cells[1]),
                "points": float(cells[-1]),
            }
        )
    if kind == "team":
        html = Path(path).read_text(encoding="utf-8", errors="ignore")
        for raw in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)[1:]:
            cells = [
                unescape(c).strip() for c in re.sub(r"<[^>]+>", "\t", raw).split("\t") if c.strip()
            ]
            if len(cells) >= 3 and _INTEGER.match(cells[-1]):
                out.append({"team": cells[1], "points": float(cells[-1])})
    return out


def _constructor_from_name(name: str, season: int) -> str | None:
    aliases = constructor_aliases_for_season(season)
    if name in aliases:
        return aliases[name]
    folded = _norm(name)
    for pattern, constructor in (
        (r"racing bulls|\brb\b|visa cash app rb", "rb"),
        (r"red bull", "red_bull"),
        (r"alphatauri", "alpha_tauri"),
        (r"alfa romeo", "alfa_romeo"),
        (r"kick sauber|stake f1|sauber", "sauber"),
        (r"aston martin", "aston_martin"),
        (r"mercedes", "mercedes"),
        (r"ferrari", "ferrari"),
        (r"mclaren", "mclaren"),
        (r"alpine", "alpine"),
        (r"williams", "williams"),
        (r"haas", "haas"),
        (r"audi", "audi"),
        (r"cadillac", "cadillac"),
    ):
        if re.search(pattern, folded):
            return constructor
    return None


# --------------------------------------------------------------------- rules


def _race_scale(season: int) -> dict[int, int]:
    return {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


REDUCED = {
    "reduced_col1_2laps_to_lt25pct": {1: 6, 2: 4, 3: 3, 4: 2, 5: 1},
    "reduced_col2_25_to_lt50pct": {1: 13, 2: 10, 3: 8, 4: 6, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    "reduced_col3_50_to_lt75pct": {1: 19, 2: 14, 3: 12, 4: 10, 5: 8, 6: 6, 7: 4, 8: 3, 9: 2, 10: 1},
}
SPRINT_SCALE = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def _fastest_lap_point_in_force(season: int) -> bool:
    return season <= 2024


def _position(token: str | None) -> tuple[int | None, str | None, bool]:
    if not token:
        return None, None, False
    fastest = "F" in token.split() or bool(re.search(r"\dF|F\d", token.replace(" ", "")))
    clean = token.replace("F", "").strip()
    if re.fullmatch(r"\d{1,2}", clean):
        return int(clean), None, fastest
    return None, clean or None, fastest


# --------------------------------------------------------------------- build


def _doc_key(doc: dict[str, Any]) -> str:
    if doc.get("retained"):
        return doc["retained"]["sha256"]
    return f"unretained:{doc['event_id']}:{doc['document_id']}:{doc['publication_cet']}"


def _compact(value: Any) -> Any:
    """Replace embedded document references by their key; full references live in 'documents'."""
    if isinstance(value, dict):
        if "document_key" in value and "registry_publication_cet" in value:
            return value["document_key"]
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def _doc_ref(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_key": _doc_key(doc),
        "title": doc["title"],
        "registry_document_id": doc["document_id"],
        "cover_document_number": doc["parsed"].get("cover_document_number")
        if doc.get("parsed")
        else None,
        "url": doc["url"],
        "sha256": doc["retained"]["sha256"] if doc.get("retained") else None,
        "registry_publication_cet": doc["publication_cet"],
        "published_at_utc_upper_bound": doc["published_at_utc_upper_bound"],
        "cover_local_date": doc["parsed"].get("cover_local_date") if doc.get("parsed") else None,
        "cover_local_time": doc["parsed"].get("cover_local_time") if doc.get("parsed") else None,
        "listed_on_event_id": doc["event_id"],
        "document_type": doc["document_type"],
        "recalled": doc["recalled"],
    }


def build(root: Path, collection_path: Path) -> dict[str, Any]:
    root = root.resolve()
    collection_bytes = collection_path.read_bytes()
    collection = json.loads(collection_bytes)
    gold = _load_gold_identities(root)
    records: list[dict[str, Any]] = []
    constructor_records: list[dict[str, Any]] = []
    documents_out: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    season_checks: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    for season_key in sorted(collection["seasons"]):
        season = int(season_key)
        season_data = collection["seasons"][season_key]
        schedule = json.loads(
            Path(root / season_data["jolpica_schedule"]["path"]).read_text(encoding="utf-8")
        )
        calendar = [
            {"round": int(race["round"]), "event_name": race["raceName"], "date": race["date"]}
            for race in schedule["MRData"]["RaceTable"]["Races"]
        ]
        events = {event["round"]: event for event in season_data["events"] if event["completed"]}
        # Every FIA points document listed on any event page of the season.
        docs: list[dict[str, Any]] = []
        for event in season_data["events"]:
            for row in event.get("documents", []):
                if row.get("kind") != "points":
                    continue
                doc = dict(row)
                doc["event_id"] = event["event_id"]
                doc["event_round"] = event["round"]
                doc["document_type"] = (
                    "championship_points_after_sprint"
                    if re.search(r"after sprint", row["title"], re.I)
                    else "championship_points"
                )
                doc["published_at_utc_upper_bound"] = _publication_upper_bound(
                    row["publication_cet"]
                ).isoformat()
                doc["parsed"] = None
                if row.get("retained"):
                    content = (root / row["retained"]["path"]).read_bytes()
                    if hashlib.sha256(content).hexdigest() != row["retained"]["sha256"]:
                        raise ValueError("retained FIA document bytes do not match their hash")
                    try:
                        doc["parsed"] = parse_points_pdf(content)
                        doc["column_rounds"] = {
                            str(k): v for k, v in _align(doc["parsed"]["headers"], calendar).items()
                        }
                    except ValueError as error:
                        doc["parse_error"] = str(error)
                docs.append(doc)
        docs.sort(
            key=lambda d: (
                d["published_at_utc_upper_bound"],
                int(d["parsed"]["cover_document_number"])
                if d.get("parsed") and d["parsed"].get("cover_document_number")
                else 0,
                d["url"] or "",
            )
        )
        for doc in docs:
            entry = _doc_ref(doc)
            entry["season"] = season
            if doc.get("parsed"):
                parsed = doc["parsed"]
                entry["parser_version"] = PARSER_VERSION
                entry["headers"] = parsed["headers"]
                entry["column_rounds"] = doc["column_rounds"]
                entry["driver_rows"] = len(parsed["drivers"])
                entry["entrant_rows"] = len(parsed["entrants"])
                entry["notes"] = parsed["notes"]
                entry["rows_where_printed_total_differs_from_cell_sum"] = [
                    {
                        "kind": kind,
                        "name": row["name"],
                        "printed_total": row["total"],
                        "sum_of_cells": row["sum_of_cells"],
                    }
                    for kind in ("drivers", "entrants")
                    for row in parsed[kind]
                    if not row["total_matches_sum"]
                ]
                for item in entry["rows_where_printed_total_differs_from_cell_sum"]:
                    conflicts.append(
                        {
                            "season": season,
                            "event_id": doc["event_id"],
                            "type": "fia_document_internal_total_mismatch",
                            "detail": item,
                            "document": _doc_ref(doc),
                            "resolution": (
                                "per-event cells are used; printed TOTAL is not used for event "
                                "points"
                            ),
                        }
                    )
                # Column data must belong to events raced before publication.
                for column, round_number in doc["column_rounds"].items():
                    has_data = any(column in row["positions"] for row in parsed["drivers"])
                    race_day = next(r["date"] for r in calendar if r["round"] == round_number)
                    # An after-sprint document precedes its own Grand Prix.
                    own_sprint = (
                        doc["document_type"] == "championship_points_after_sprint"
                        and round_number == doc["event_round"]
                    )
                    if (
                        has_data
                        and not own_sprint
                        and race_day > doc["published_at_utc_upper_bound"][:10]
                    ):
                        raise ValueError(f"{doc['url']} carries data for a future event")
                unmatched = [
                    index
                    for index in range(len(parsed["headers"]))
                    if str(index) not in doc["column_rounds"]
                    and any(str(index) in row["positions"] for row in parsed["drivers"])
                ]
                if unmatched:
                    raise ValueError(f"{doc['url']} has data in an unaligned column")
            elif doc.get("parse_error"):
                entry["parse_error"] = doc["parse_error"]
            documents_out.append(entry)
        for round_number in sorted(events):
            event = events[round_number]
            outputs = _build_event(season, event, docs, gold, root)
            records.extend(outputs["records"])
            constructor_records.extend(outputs["constructors"])
            conflicts.extend(outputs["conflicts"])
            if outputs["unresolved"]:
                unresolved.append(outputs["unresolved"])
            event_summaries.append(outputs["summary"])
        season_checks.extend(_season_checks(season, season_data, docs, root, records))
    for item in season_checks:
        if not item["agrees"]:
            conflicts.append(
                {"season": item["season"], "type": "season_total_mismatch", "detail": item}
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "audit_as_of_utc": AUDIT_AS_OF.isoformat(),
        "collection_index": {
            "path": collection_path.resolve().relative_to(root).as_posix(),
            "sha256": hashlib.sha256(collection_bytes).hexdigest(),
        },
        "field_definitions": FIELD_DEFINITIONS,
        "evidence_quality_levels": EVIDENCE_QUALITY,
        "records": _compact(
            sorted(
                records,
                key=lambda r: (r["season"], r["round"], r["driver"] or "", r["fia_driver_name"]),
            )
        ),
        "constructor_event_points": _compact(
            sorted(
                constructor_records, key=lambda r: (r["season"], r["round"], r["constructor"] or "")
            )
        ),
        "conflicts_note": (
            "Document references inside records, constructor_event_points and conflicts are "
            "document_key values; resolve them through 'documents'."
        ),
        "event_summaries": event_summaries,
        "season_total_checks": season_checks,
        "conflicts": _compact(conflicts),
        "unresolved_events": unresolved,
        "documents": documents_out,
    }
    return payload


def _load_gold_identities(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    paths = sorted(root.glob("data/benchmarks/gold_historical_enrichment_v2/*/*/*/gold.parquet"))
    if not paths:
        return {}
    import pandas as pd

    frame = pd.read_parquet(
        paths[0],
        columns=[
            "event_id",
            "driver_id",
            "constructor_id",
            "label_position",
            "label_audit_reference",
        ],
    )
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        out[row.event_id][row.driver_id] = {
            "constructor_id": row.constructor_id,
            "label_position": None if pd.isna(row.label_position) else int(row.label_position),
            "label_audit_reference": row.label_audit_reference,
        }
    out["__source__"] = {
        "path": {
            "path": paths[0].relative_to(root).as_posix(),
            "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        }
    }
    return out


def _cell(doc: dict[str, Any], round_number: int) -> str | None:
    for column, mapped in doc.get("column_rounds", {}).items():
        if mapped == round_number:
            return column
    return None


def _driver_key(name: str) -> str:
    """FIA labels vary between documents ('A. ANTONELLI', 'K. ANTONELLI')."""
    return driver_from_name(name) or f"unmapped:{name}"


def _driver_values(doc: dict[str, Any], column: str) -> dict[str, dict[str, Any]]:
    values = {}
    for row in doc["parsed"]["drivers"]:
        # After-sprint documents print sprint points without a position, and a
        # driver who scored in a sprint but did not start the race has no race
        # position; both still carry a points value for the event.
        if column not in row["positions"] and column not in row["points"]:
            continue
        values[_driver_key(row["name"])] = {
            "points": row["points"].get(column, 0.0),
            "position_token": row["positions"].get(column),
            "name": row["name"],
        }
    return values


def _entrant_values(doc: dict[str, Any], column: str) -> dict[str, float]:
    return {
        row["name"]: row["points"].get(column, 0.0)
        for row in doc["parsed"]["entrants"]
        if column in row["positions"] or column in row["points"]
    }


def _build_event(
    season: int,
    event: dict[str, Any],
    docs: list[dict[str, Any]],
    gold: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    round_number = event["round"]
    event_id = event["event_id"]
    conflicts: list[dict[str, Any]] = []
    own = [d for d in docs if d["event_round"] == round_number]
    own_sprint = [
        d
        for d in own
        if d["document_type"] == "championship_points_after_sprint" and d.get("parsed")
    ]
    recalled = [d for d in own if d["recalled"]]
    # Combined weekend values: every parsed document carrying this event's
    # column, except this event's own after-sprint documents.
    combined_docs = [
        d
        for d in docs
        if d.get("parsed")
        and _cell(d, round_number) is not None
        and not (
            d["event_round"] == round_number
            and d["document_type"] == "championship_points_after_sprint"
        )
        and _driver_values(d, _cell(d, round_number))
    ]
    summary: dict[str, Any] = {
        "season": season,
        "round": round_number,
        "event_id": event_id,
        "event": event["event_name"],
        "has_sprint": event["has_sprint"],
        "in_gold": event_id in gold,
        "own_points_documents": [_doc_ref(d) for d in own],
        "recalled_points_documents": [_doc_ref(d) for d in recalled],
        "classification_documents": [
            {
                "title": row["title"],
                "registry_document_id": row["document_id"],
                "url": row["url"],
                "registry_publication_cet": row["publication_cet"],
                "recalled": row["recalled"],
            }
            for row in event.get("classification_documents", [])
        ],
        "review_documents": [
            {
                "title": row["title"],
                "registry_document_id": row["document_id"],
                "url": row["url"],
                "sha256": row["retained"]["sha256"] if row.get("retained") else None,
                "registry_publication_cet": row["publication_cet"],
            }
            for row in event.get("documents", [])
            if row.get("kind") == "review"
        ],
    }
    if not combined_docs:
        summary["status"] = "unresolved_no_fia_points_matrix"
        return {
            "records": [],
            "constructors": [],
            "conflicts": [],
            "unresolved": {
                "event_id": event_id,
                "event": event["event_name"],
                "reason": "no FIA championship points document carries this event",
            },
            "summary": summary,
        }
    first_doc = combined_docs[0]
    final_doc = combined_docs[-1]
    own_combined = [d for d in combined_docs if d["event_round"] == round_number]
    summary["first_fia_points_document"] = _doc_ref(first_doc)
    summary["latest_fia_points_document"] = _doc_ref(final_doc)
    summary["event_points_published_in_own_document"] = bool(own_combined)
    final_column = _cell(final_doc, round_number)
    final_values = _driver_values(final_doc, final_column)
    sprint_values: dict[str, Any] = {}
    sprint_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in own_sprint:
        values = _driver_values(doc, _cell(doc, round_number))
        for name, value in values.items():
            sprint_history[name].append(
                {
                    "document": _doc_ref(doc),
                    "points": value["points"],
                    "position_token": value["position_token"],
                }
            )
        # A driver listed in the after-sprint matrix but without a cell in the
        # sprint column scored no sprint points there.
        for row in doc["parsed"]["drivers"]:
            if _driver_key(row["name"]) not in values:
                values[_driver_key(row["name"])] = {
                    "points": 0.0,
                    "position_token": None,
                    "name": row["name"],
                }
                sprint_history[_driver_key(row["name"])].append(
                    {"document": _doc_ref(doc), "points": 0.0, "position_token": None}
                )
        sprint_values = values  # latest after-sprint document wins
    # Secondary and tertiary sources.
    f1_race = {}
    f1_sprint = {}
    if event.get("formula1"):
        f1_race = {
            row["driver"]: row for row in _f1_result_rows(root / event["formula1"]["race"]["path"])
        }
        if event["formula1"].get("sprint"):
            f1_sprint = {
                row["driver"]: row
                for row in _f1_result_rows(root / event["formula1"]["sprint"]["path"])
            }
    jolpica_race = {}
    jolpica_sprint = {}
    for kind, target in (("results", jolpica_race), ("sprint", jolpica_sprint)):
        record = event.get(f"jolpica_{kind}")
        if not record:
            continue
        races = json.loads((root / record["path"]).read_text(encoding="utf-8"))["MRData"][
            "RaceTable"
        ]["Races"]
        key = "Results" if kind == "results" else "SprintResults"
        for race in races:
            for row in race.get(key, []):
                driver = driver_from_name(row["Driver"]["familyName"])
                target[driver] = {
                    "points": float(row["points"]),
                    "position_text": row["positionText"],
                    "constructor": JOLPICA_CONSTRUCTORS.get(
                        row["Constructor"]["constructorId"], row["Constructor"]["constructorId"]
                    ),
                }
    gold_event = gold.get(event_id, {})
    pending_notes = [
        note
        for note in final_doc["parsed"]["notes"]
        if re.search(
            rf"\({re.escape(event['fia_event_name'])}\)|\({re.escape(event['event_name'])}\)", note
        )
    ]
    records = []
    race_scale = _race_scale(season)
    reduced_fits: dict[str, int] = defaultdict(int)
    # The FIA matrix leaves some non-classified retirements blank. Every row's
    # cells reconcile to its printed total, so a blank cell for a listed driver
    # who took part in the event (per formula1.com or Jolpica) is zero points.
    gold_event = gold.get(event_id, {})
    participants = set(f1_race) | set(jolpica_race) | set(gold_event)
    blank_cells = set()
    for row in final_doc["parsed"]["drivers"]:
        key = _driver_key(row["name"])
        if key not in final_values and key in participants:
            final_values[key] = {"points": 0.0, "position_token": None, "name": row["name"]}
            blank_cells.add(key)
    for name, value in sorted(final_values.items()):
        driver = driver_from_name(value["name"])
        history = []
        for doc in combined_docs:
            values = _driver_values(doc, _cell(doc, round_number))
            listed = any(_driver_key(row["name"]) == name for row in doc["parsed"]["drivers"])
            if name in values:
                history.append(
                    {
                        "document": _doc_ref(doc),
                        "points": values[name]["points"],
                        "position_token": values[name]["position_token"],
                    }
                )
            elif name in blank_cells and listed:
                history.append(
                    {
                        "document": _doc_ref(doc),
                        "points": 0.0,
                        "position_token": None,
                        "blank_cell": True,
                    }
                )
        total = value["points"]
        changes = [
            {
                "from_points": previous["points"],
                "to_points": current["points"],
                "document": current["document"],
            }
            for previous, current in zip(history, history[1:], strict=False)
            if previous["points"] != current["points"]
        ]
        stable_from = history[-1]
        for item in reversed(history):
            if item["points"] != total:
                break
            stable_from = item
        position, status, fastest = _position(value["position_token"])
        # Sprint split.
        sprint_points = None
        sprint_source = None
        sprint_position_token = None
        if event["has_sprint"]:
            if name in sprint_values:
                sprint_points = sprint_values[name]["points"]
                sprint_position_token = sprint_values[name]["position_token"]
                sprint_source = "fia_championship_points_after_sprint"
            elif own_sprint and driver in f1_sprint and f1_sprint[driver]["points"] is not None:
                # The FIA after-sprint document omits this driver (e.g. a missing page).
                sprint_points = f1_sprint[driver]["points"]
                sprint_position_token = f1_sprint[driver]["position_text"]
                sprint_source = (
                    "formula1_com_sprint_results_driver_absent_from_fia_after_sprint_document"
                )
            elif own_sprint:
                sprint_points = None
                sprint_source = "driver_absent_from_fia_after_sprint_document"
            elif driver in f1_sprint and f1_sprint[driver]["points"] is not None:
                sprint_points = f1_sprint[driver]["points"]
                sprint_position_token = f1_sprint[driver]["position_text"]
                sprint_source = "formula1_com_sprint_results"
            if sprint_points is None and total == 0:
                # Points are non-negative, so a zero weekend total fixes both parts.
                sprint_points = 0.0
                sprint_source = "fia_weekend_total_zero"
        else:
            sprint_points = 0.0
            sprint_source = "no_sprint_at_event"
        race_including_bonus = None if sprint_points is None else total - sprint_points
        bonus = None
        race_points = None
        checks: dict[str, Any] = {}
        if race_including_bonus is not None:
            expected_full = race_scale.get(position, 0) if position else 0
            bonus = 0.0
            if (
                _fastest_lap_point_in_force(season)
                and fastest
                and position
                and position <= 10
                and race_including_bonus - expected_full == 1
            ):
                bonus = 1.0
            race_points = race_including_bonus - bonus
            checks["race_points_match_full_scale"] = race_points == expected_full
            if not checks["race_points_match_full_scale"]:
                for label, scale in REDUCED.items():
                    if race_points == (scale.get(position, 0) if position else 0):
                        reduced_fits[label] += 1
            if (
                _fastest_lap_point_in_force(season)
                and fastest
                and position
                and position <= 10
                and bonus == 0
            ):
                checks["fastest_lap_marker_without_bonus_point"] = True
        sprint_position_source = (
            "fia_championship_points_after_sprint" if sprint_position_token else None
        )
        if event["has_sprint"] and sprint_position_token is None and driver in f1_sprint:
            # FIA after-sprint matrices print sprint points without positions.
            sprint_position_token = f1_sprint[driver]["position_text"]
            sprint_position_source = "formula1_com_sprint_results"
        if event["has_sprint"] and sprint_points is not None:
            sprint_position, _status, _ = _position(sprint_position_token)
            checks["sprint_points_match_scale"] = sprint_points == SPRINT_SCALE.get(
                sprint_position or 0, 0
            )
        # Cross-checks against formula1.com and Jolpica (current state).
        f1_row = f1_race.get(driver)
        if f1_row is not None and race_including_bonus is not None:
            checks["formula1_race_points"] = f1_row["points"]
            checks["formula1_race_agrees"] = f1_row["points"] == race_including_bonus
        if (
            event["has_sprint"]
            and driver in f1_sprint
            and not (sprint_source or "").startswith("formula1_com_sprint_results")
        ):
            checks["formula1_sprint_points"] = f1_sprint[driver]["points"]
            checks["formula1_sprint_agrees"] = f1_sprint[driver]["points"] == sprint_points
        if driver in jolpica_race and race_including_bonus is not None:
            checks["jolpica_race_points"] = jolpica_race[driver]["points"]
            checks["jolpica_race_agrees"] = jolpica_race[driver]["points"] == race_including_bonus
        if driver in jolpica_sprint and sprint_points is not None:
            checks["jolpica_sprint_points"] = jolpica_sprint[driver]["points"]
            checks["jolpica_sprint_agrees"] = jolpica_sprint[driver]["points"] == sprint_points
        gold_row = gold_event.get(driver) if isinstance(gold_event, dict) else None
        if gold_row:
            checks["gold_label_position"] = gold_row["label_position"]
            if position is not None and gold_row["label_position"] is not None:
                checks["gold_position_agrees"] = gold_row["label_position"] == position
        # Constructor identity: audited Gold roster, else formula1.com, else Jolpica.
        constructor, constructor_source = None, None
        if gold_row:
            constructor, constructor_source = gold_row["constructor_id"], "gold_roster"
        elif f1_row is not None:
            constructor = _constructor_from_name(f1_row["team"], season)
            constructor_source = "formula1_com_race_result_team" if constructor else None
        if constructor is None and driver in jolpica_race:
            constructor, constructor_source = jolpica_race[driver]["constructor"], "jolpica_results"
        if (
            f1_row is not None
            and constructor
            and _constructor_from_name(f1_row["team"], season) not in {None, constructor}
        ):
            checks["formula1_constructor_disagrees"] = _constructor_from_name(
                f1_row["team"], season
            )
        adjustments = []
        if name in blank_cells:
            adjustments.append(
                {
                    "type": "fia_blank_cell_read_as_zero",
                    "detail": (
                        "driver listed in the FIA matrix with no cell for this event; took part "
                        "per secondary sources"
                    ),
                }
            )
            checks["fia_cell_blank"] = True
        if status in {"DQ", "DSQ", "EX"}:
            adjustments.append({"type": "race_classification_status", "status": status})
        if sprint_position_token and _position(sprint_position_token)[1] in {"DQ", "DSQ", "EX"}:
            adjustments.append(
                {
                    "type": "sprint_classification_status",
                    "status": _position(sprint_position_token)[1],
                }
            )
        for change in changes:
            adjustments.append({"type": "championship_points_revised", **change})
        if len(sprint_history.get(name, [])) > 1:
            for previous, current in zip(
                sprint_history[name], sprint_history[name][1:], strict=False
            ):
                if previous["points"] != current["points"]:
                    adjustments.append(
                        {
                            "type": "sprint_points_revised",
                            "from_points": previous["points"],
                            "to_points": current["points"],
                            "document": current["document"],
                        }
                    )
        for note in pending_notes:
            adjustments.append(
                {
                    "type": "pending_appeal_note_on_latest_document",
                    "text": note,
                    "document": _doc_ref(final_doc),
                }
            )
        disagreements = [
            key for key, agreed in checks.items() if key.endswith("_agrees") and agreed is False
        ]
        formula1_confirmed = checks.get("formula1_race_agrees") is True and (
            not event["has_sprint"]
            or checks.get(
                "formula1_sprint_agrees",
                (sprint_source or "").startswith("formula1_com_sprint_results"),
            )
        )
        if driver is None:
            quality = "conflict_unmapped_driver"
        elif sprint_points is None:
            quality = "unknown_sprint_split"
        elif any(key.startswith("formula1") for key in disagreements):
            quality = "conflict_secondary_source"
        elif not own_combined:
            quality = "fia_later_document_only" + (
                "_formula1_confirmed" if formula1_confirmed else ""
            )
        elif sprint_source and sprint_source.startswith("formula1_com_sprint_results"):
            quality = (
                "fia_total_formula1_sprint_split"
                if checks.get("formula1_race_agrees")
                else "fia_total_unconfirmed_sprint_split"
            )
        elif formula1_confirmed:
            quality = "fia_official_formula1_confirmed"
        else:
            quality = "fia_official_unconfirmed"
        if pending_notes:
            revision_status = "pending_appeal"
        elif changes or any(a["type"] == "sprint_points_revised" for a in adjustments):
            revision_status = "revised"
        else:
            revision_status = "unrevised"
        record = {
            "season": season,
            "round": round_number,
            "event": event["event_name"],
            "event_id_if_known": event_id,
            "in_gold": bool(gold_row),
            "driver": driver,
            "fia_driver_name": value["name"],
            "constructor": constructor,
            "constructor_source": constructor_source,
            "race_points": race_points,
            "sprint_points": sprint_points,
            "bonus_points": bonus,
            "penalties_or_adjustments": adjustments,
            "total_event_points": total,
            "race_position_fia": value["position_token"],
            "race_fastest_lap_marker": fastest,
            "sprint_position": sprint_position_token,
            "sprint_position_source": sprint_position_source,
            "classification_source": {
                "race_points_matrix_position_cell": "fia_championship_points_document",
                "fia_classification_documents": (
                    "see event_summaries[event_id].classification_documents"
                ),
                "gold_label_audit_reference": gold_row["label_audit_reference"]
                if gold_row
                else None,
            },
            "scoring_source": {
                "total": _doc_ref(final_doc),
                "sprint_split": sprint_source,
                "sprint_document": sprint_history[name][-1]["document"]
                if sprint_history.get(name)
                else None,
                "fastest_lap_bonus_rule": "scoring_rules.json fastest_lap_rule for season",
            },
            "published_at": stable_from["document"]["published_at_utc_upper_bound"],
            "first_published_at": history[0]["document"]["published_at_utc_upper_bound"],
            # Value as of time t = last timeline entry published at or before t.
            "points_timeline": [
                {
                    "document": item["document"],
                    "published_at_utc_upper_bound": item["document"][
                        "published_at_utc_upper_bound"
                    ],
                    "points": item["points"],
                    "position_token": item["position_token"],
                }
                for index, item in enumerate(history)
                if index == 0 or item["points"] != history[index - 1]["points"]
            ],
            "fia_documents_carrying_value": len(history),
            "sprint_history": sprint_history.get(name, []),
            "revision_status": revision_status,
            "evidence_quality": quality,
            "checks": checks,
        }
        for key in disagreements:
            conflicts.append(
                {
                    "season": season,
                    "event_id": event_id,
                    "driver": driver,
                    "type": "source_disagreement",
                    "check": key,
                    "fia_total_event_points": total,
                    "fia_sprint_points": sprint_points,
                    "detail": {k: v for k, v in checks.items() if not k.endswith("_agrees")},
                }
            )
        records.append(record)
    # Drivers in secondary sources but absent from the FIA matrix column.
    mapped = {record["driver"] for record in records}
    for driver in sorted(set(f1_race) - mapped - {None}):
        conflicts.append(
            {
                "season": season,
                "event_id": event_id,
                "driver": driver,
                "type": "driver_missing_from_fia_matrix_column",
            }
        )
    for record in records:
        if record["driver"] is None:
            conflicts.append(
                {
                    "season": season,
                    "event_id": event_id,
                    "type": "unmapped_fia_driver_name",
                    "name": record["fia_driver_name"],
                }
            )
    # Constructors: sum of drivers versus the FIA entrant matrix.
    constructors = []
    entrant_values = _entrant_values(final_doc, final_column)
    sums: dict[str, float] = defaultdict(float)
    sprint_sums: dict[str, float | None] = {}
    for record in records:
        sums[record["constructor"]] += record["total_event_points"]
        if record["sprint_points"] is None or sprint_sums.get(record["constructor"], 0.0) is None:
            sprint_sums[record["constructor"]] = None
        else:
            sprint_sums[record["constructor"]] = (
                sprint_sums.get(record["constructor"], 0.0) + record["sprint_points"]
            )
    entrant_by_constructor = {}
    for entrant, points in entrant_values.items():
        constructor = _constructor_from_name(entrant, season)
        entrant_by_constructor[constructor] = (entrant, points)
    for constructor in sorted(set(sums) | set(entrant_by_constructor), key=lambda c: c or ""):
        entrant, fia_points = entrant_by_constructor.get(constructor, (None, None))
        agrees = fia_points is not None and fia_points == sums.get(constructor)
        constructors.append(
            {
                "season": season,
                "round": round_number,
                "event": event["event_name"],
                "event_id_if_known": event_id,
                "constructor": constructor,
                "fia_entrant_name": entrant,
                "fia_entrant_event_points": fia_points,
                "sum_of_driver_event_points": sums.get(constructor),
                "sum_of_driver_sprint_points": sprint_sums.get(constructor),
                "agrees": agrees,
                "source": _doc_ref(final_doc),
            }
        )
        if not agrees:
            conflicts.append(
                {
                    "season": season,
                    "event_id": event_id,
                    "type": "constructor_sum_mismatch",
                    "constructor": constructor,
                    "fia_entrant_name": entrant,
                    "fia_entrant_event_points": fia_points,
                    "sum_of_driver_event_points": sums.get(constructor),
                }
            )
    expected_fast = sum(1 for r in records if r["bonus_points"] == 1)
    mismatched = [r for r in records if r["checks"].get("race_points_match_full_scale") is False]
    summary.update(
        {
            "status": "resolved",
            "drivers": len(records),
            "fastest_lap_bonus_awarded": expected_fast,
            "drivers_not_matching_full_race_scale": [
                {
                    "driver": r["driver"],
                    "position": r["race_position_fia"],
                    "race_points": r["race_points"],
                }
                for r in mismatched
            ],
            "reduced_scale_fits": dict(reduced_fits),
            "pending_appeal_notes": pending_notes,
            "revisions": sorted(
                {
                    (a["document"]["url"] or "", a["document"]["registry_publication_cet"])
                    for r in records
                    for a in r["penalties_or_adjustments"]
                    if a["type"].endswith("_revised")
                }
            ),
        }
    )
    for r in mismatched:
        conflicts.append(
            {
                "season": season,
                "event_id": event_id,
                "driver": r["driver"],
                "type": "race_points_not_on_full_scale",
                "race_position_fia": r["race_position_fia"],
                "race_points": r["race_points"],
                "bonus_points": r["bonus_points"],
            }
        )
    return {
        "records": records,
        "constructors": constructors,
        "conflicts": conflicts,
        "unresolved": None,
        "summary": summary,
    }


def _season_checks(
    season: int,
    season_data: dict[str, Any],
    docs: list[dict[str, Any]],
    root: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed = [d for d in docs if d.get("parsed")]
    if not parsed:
        return []
    latest = parsed[-1]
    out = []
    f1_drivers = {
        row["driver"]: row["points"]
        for row in _f1_standings(
            root / season_data["formula1_standings"]["drivers"]["path"], "drivers"
        )
    }
    ledger: dict[str, float] = defaultdict(float)
    for record in records:
        if record["season"] == season and record["driver"]:
            ledger[record["driver"]] += record["total_event_points"]
    for row in latest["parsed"]["drivers"]:
        driver = driver_from_name(row["name"])
        out.append(
            {
                "season": season,
                "kind": "driver",
                "entity": driver,
                "fia_latest_document_printed_total": row["total"],
                "fia_latest_document_cell_sum": row["sum_of_cells"],
                "ledger_sum_of_event_records": ledger.get(driver),
                "formula1_com_standings_total": f1_drivers.get(driver),
                "agrees": row["sum_of_cells"] == ledger.get(driver, 0.0)
                and f1_drivers.get(driver) in {row["sum_of_cells"], None}
                and row["total"] == row["sum_of_cells"],
                "fia_document": _doc_ref(latest),
            }
        )
    f1_teams = {
        _constructor_from_name(row["team"], season): row["points"]
        for row in _f1_standings(root / season_data["formula1_standings"]["team"]["path"], "team")
    }
    for row in latest["parsed"]["entrants"]:
        constructor = _constructor_from_name(row["name"], season)
        out.append(
            {
                "season": season,
                "kind": "constructor",
                "entity": constructor,
                "fia_entrant_name": row["name"],
                "fia_latest_document_printed_total": row["total"],
                "fia_latest_document_cell_sum": row["sum_of_cells"],
                "formula1_com_standings_total": f1_teams.get(constructor),
                "agrees": f1_teams.get(constructor) in {row["total"], None}
                and row["total"] == row["sum_of_cells"],
                "fia_document": _doc_ref(latest),
            }
        )
    return out


# Curated from visual inspection of each retained regulation issue. Pink or red
# revision text was read as shown in the PDF: struck-through text is deleted.
_FL_2022_2024 = (
    "One point to the driver with the fastest valid race lap and to that car's constructor, "
    "only if the driver is classified in the top ten of the final race classification and the "
    "leader completed at least 50% of the scheduled race distance."
)
_FL_NONE = "No fastest-lap point."
_REDUCED_2022 = (
    "Applies only if the race is suspended under Article 57 and cannot be resumed. Less than two "
    "laps by the leader: no points. Otherwise no points unless two laps were completed without SC "
    "or VSC; 2 laps to <25% of scheduled distance: column 1; 25% to <50%: column 2; 50% to "
    "<75%: column 3; 75% or more: full points (Article 6.4). A suspended race that resumes "
    "receives full points."
)
_REDUCED_2023_2025 = (
    "Applies whenever the race distance from the start signal to the end-of-session signal is "
    "less than the scheduled race distance. Less than two laps by the leader: no points. Otherwise "
    "no points unless two laps were completed without SC or VSC; 2 laps to <25%: column 1; 25% "
    "to <50%: column 2; 50% to <75%: column 3; 75% or more: full points."
)
_REDUCED_2026 = (
    "Article A2.2.1: points depend on the laps completed by the leader between the start signal "
    "and the end-of-session signal; in all cases no points unless two complete consecutive laps "
    "were completed by the leader without SC or VSC. 2 laps to <25%: column 1; 25% to <50%: "
    "column 2; 50% to <75%: column 3; 75% or more: column 4 (full scale)."
)
_SPRINT_REDUCED_2022 = (
    "Article 6.6: if the sprint is suspended and cannot be resumed, no points if the leader has "
    "not completed two laps without SC or VSC or has completed less than 50% of the scheduled "
    "sprint distance."
)
_SPRINT_REDUCED_2023_2025 = (
    "Article 6.6: if the sprint distance is less than scheduled, no points unless two laps were "
    "completed by the leader without SC or VSC; less than 50% of scheduled sprint distance: no "
    "points; 50% or more: full sprint points."
)
_SPRINT_REDUCED_2026 = (
    "Article A2.2.2: no points unless two complete consecutive laps without SC or VSC; less "
    "than 50% of the scheduled sprint distance: no points; 50% or more: full sprint scale."
)
RULE_PERIODS = [
    {
        "season": 2022,
        "effective_from": "2022-03-15",
        "effective_to": "2022-12-31",
        "first_issue_in_force": "2022 - iss 5 - 2022-03-15",
        "last_issue_checked": "Issue 9 - 2022-10-19",
        "articles": "Articles 6.4, 6.5, 6.6, 7.1, 7.2",
        "fastest_lap_rule": _FL_2022_2024,
        "reduced_points_rule": _REDUCED_2022,
        "sprint_reduced_rule": _SPRINT_REDUCED_2022,
        "reduced_columns": REDUCED,
    },
    {
        "season": 2023,
        "effective_from": "2023-02-22",
        "effective_to": "2023-12-31",
        "first_issue_in_force": "2023 Formula 1 Sporting Regulations - Issue 4 - 2023-02-22",
        "last_issue_checked": "2023 Formula 1 Sporting Regulations - Issue 8 - 2023-12-06",
        "articles": "Articles 6.4, 6.5, 6.6, 7.1, 7.2",
        "fastest_lap_rule": _FL_2022_2024,
        "reduced_points_rule": _REDUCED_2023_2025,
        "sprint_reduced_rule": _SPRINT_REDUCED_2023_2025,
        "reduced_columns": REDUCED,
    },
    {
        "season": 2024,
        "effective_from": "2023-09-26",
        "effective_to": "2024-12-31",
        "first_issue_in_force": "2024 Formula 1 Sporting Regulations - Issue 1 - 2023-09-26",
        "last_issue_checked": "2024 Formula 1 Sporting Regulations - Issue 7 - 2024-07-31",
        "articles": "Articles 6.4, 6.5, 6.6, 7.1, 7.2",
        "fastest_lap_rule": _FL_2022_2024,
        "reduced_points_rule": _REDUCED_2023_2025,
        "sprint_reduced_rule": _SPRINT_REDUCED_2023_2025,
        "reduced_columns": REDUCED,
    },
    {
        "season": 2025,
        "effective_from": "2024-10-17",
        "effective_to": "2025-12-31",
        "first_issue_in_force": "2025 Formula 1 Sporting Regulations - Issue 2- 2024-10-17",
        "last_issue_checked": "2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30",
        "articles": "Articles 6.4, 6.5, 6.6, 7.1, 7.2",
        "fastest_lap_rule": _FL_NONE,
        "reduced_points_rule": _REDUCED_2023_2025,
        "sprint_reduced_rule": _SPRINT_REDUCED_2023_2025,
        "reduced_columns": REDUCED,
    },
    {
        "season": 2026,
        "effective_from": "2026-02-27",
        "effective_to": "2026-12-31",
        "first_issue_in_force": "Section A [General Provisions] - Iss 02 - 2026-02-27",
        "last_issue_checked": "Section A [General Provisions] - Iss 03 - 2026-06-25",
        "articles": "Section A Articles A2.1.4c, A2.2.1, A2.2.2, A2.2.3",
        "fastest_lap_rule": _FL_NONE,
        "reduced_points_rule": _REDUCED_2026,
        "sprint_reduced_rule": _SPRINT_REDUCED_2026,
        "reduced_columns": REDUCED,
    },
]
SEASON_NOTES = {
    2022: {
        "constructor_scoring": (
            "Article 6.2: constructor points are the sum of the results of both cars."
        ),
        "tie_break": (
            "Article 7.2: most first places in a race, then second places, and so on; failing "
            "that the FIA nominates."
        ),
        "dead_heat": (
            "Article 7.1: points for tied positions are added together and shared equally."
        ),
        "sprint_format": (
            "Sprint session result sets the Grand Prix starting grid (Article 'grid for the race "
            "will be drawn up based on the final classification of the sprint session'). Scale "
            "8-7-6-5-4-3-2-1 for the top eight."
        ),
        "regulation_changes": [
            (
                "Issue 4 (2022-02-18) changed the sprint scale from 3-2-1 to 8-7-6-5-4-3-2-1 and "
                "introduced the reduced-points table; it was published before round 1."
            ),
            (
                "Issue 5 (2022-03-15), before round 1, corrected reduced-points column 3: 4th 9 to "
                "10 and 7th 5 to 4."
            ),
            (
                "Issues 6 to 9 contain no scoring change (wording and Event to Competition "
                "terminology only)."
            ),
        ],
    },
    2023: {
        "constructor_scoring": "Article 6.2: sum of both cars.",
        "tie_break": "Article 7.2: race-placing countback, then FIA nomination.",
        "dead_heat": "Article 7.1: shared equally.",
        "sprint_format": (
            "Sprint Shootout introduced in Issue 5 (2023-04-25), before the first 2023 sprint; "
            "the sprint no longer sets the Grand Prix grid. Sprint scale unchanged."
        ),
        "regulation_changes": [
            (
                "Issue 4 (2023-02-22), before round 1: reduced points apply to any race shorter "
                "than scheduled, no longer only to suspended races that cannot be resumed; matching"
                " sprint rule."
            ),
        ],
    },
    2024: {
        "constructor_scoring": "Article 6.2: sum of both cars.",
        "tie_break": "Article 7.2: race-placing countback, then FIA nomination.",
        "dead_heat": "Article 7.1: shared equally.",
        "sprint_format": (
            "Sprint Qualifying replaces Sprint Shootout from Issue 5 (2024-02-28). Sprint scale "
            "unchanged."
        ),
        "regulation_changes": [
            (
                "Issues 3 to 5 changed only the formation-lap-behind-safety-car time wording; "
                "scoring scales unchanged."
            ),
        ],
    },
    2025: {
        "constructor_scoring": "Article 6.2: sum of both cars.",
        "tie_break": "Article 7.2: race-placing countback, then FIA nomination.",
        "dead_heat": "Article 7.1: shared equally.",
        "sprint_format": "Sprint Qualifying format as in 2024. Sprint scale unchanged.",
        "regulation_changes": [
            (
                "Issue 2 (2024-10-17) deletes the fastest-lap point. Issues 2 to 4 still print the "
                "clause, struck through as a deletion; plain text extraction cannot see the strike-"
                "through. Issue 5 (2025-04-30) removes it entirely. No 2025 fastest-lap point was "
                "awarded in any FIA points document."
            ),
        ],
    },
    2026: {
        "constructor_scoring": "Article A2.1.4b: F1 Team points are the results of both cars.",
        "tie_break": (
            "Article A2.1.4c: race-placing countback; if still tied, the same countback is "
            "applied to the drivers' qualifying results."
        ),
        "dead_heat": "Article A2.2.3: shared equally.",
        "sprint_format": "Sprint Qualifying format; sprint scale 8-7-6-5-4-3-2-1.",
        "regulation_changes": [
            (
                "Regulations restructured into Sections A to F; championship scoring moved from the"
                " Sporting Regulations to Section A."
            ),
            (
                "Section A Issue 01 (2025-12-10) measured distance; Issue 02 (2026-02-27), before "
                "round 1, measures laps completed by the leader and applies the two-lap condition "
                "to sprints."
            ),
            "The full scale is printed as column 4 (75% or more); the scale values are unchanged.",
        ],
    },
}


def build_rules(root: Path, collection: dict[str, Any]) -> dict[str, Any]:
    regulations = collection["regulations"]["documents"]

    def source(fragment: str) -> dict[str, Any]:
        matches = [
            d
            for d in regulations
            if fragment.replace(" ", "") in d["listing_title"].replace(" ", "")
        ]
        if len(matches) != 1:
            raise ValueError(f"regulation issue {fragment!r} is missing or ambiguous")
        document = matches[0]
        content = (root / document["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != document["sha256"]:
            raise ValueError("retained regulation bytes do not match their hash")
        published = re.search(r"Published on\s*(\d{2}\.\d{2}\.\d{2,4})", document["listing_title"])
        return {
            "title": re.sub(r"\s*Published on.*$", "", document["listing_title"]),
            "url": document["url"],
            "sha256": document["sha256"],
            "listing_publication_date": published[1] if published else None,
        }

    rows = []
    periods = []
    for period in RULE_PERIODS:
        first = source(period["first_issue_in_force"])
        last = source(period["last_issue_checked"])
        rule_date = first["title"].rsplit(" - ", 1)[-1].strip()
        common = {
            "season": period["season"],
            "fastest_lap_rule": period["fastest_lap_rule"],
            "reduced_points_rule": period["reduced_points_rule"],
            "effective_from": period["effective_from"],
            "effective_to": period["effective_to"],
            "evidence_source": {
                "first_issue_in_force": first,
                "last_issue_checked": last,
                "articles": period["articles"],
            },
            "publication_or_rule_date": rule_date,
        }
        for position, points in _race_scale(period["season"]).items():
            rows.append({**common, "event_type": "race", "position": position, "points": points})
        for label, scale in period["reduced_columns"].items():
            for position, points in scale.items():
                rows.append(
                    {
                        **common,
                        "event_type": f"race_{label}",
                        "position": position,
                        "points": points,
                    }
                )
        for position, points in SPRINT_SCALE.items():
            rows.append(
                {
                    **common,
                    "event_type": "sprint",
                    "position": position,
                    "points": points,
                    "fastest_lap_rule": "No fastest-lap point in sprints.",
                    "reduced_points_rule": period["sprint_reduced_rule"],
                }
            )
        if _fastest_lap_point_in_force(period["season"]):
            rows.append(
                {**common, "event_type": "fastest_lap_bonus", "position": None, "points": 1}
            )
        periods.append(
            {
                **common,
                "sprint_reduced_rule": period["sprint_reduced_rule"],
                **SEASON_NOTES[period["season"]],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_as_of_utc": AUDIT_AS_OF.isoformat(),
        "field_definitions": {
            "season": "Championship year the rule governs.",
            "event_type": (
                "race (full scale), race_reduced_col1_2laps_to_lt25pct, "
                "race_reduced_col2_25_to_lt50pct, race_reduced_col3_50_to_lt75pct, sprint, or "
                "fastest_lap_bonus."
            ),
            "position": (
                "Classified finishing position; null for fastest_lap_bonus. Positions not listed "
                "score zero."
            ),
            "points": (
                "Points awarded to the driver and, identically, to the constructor of that car."
            ),
            "fastest_lap_rule": "Fastest-lap rule in force for the season.",
            "reduced_points_rule": (
                "When the reduced table applies (races) or when sprint points are withheld "
                "(sprints)."
            ),
            "effective_from": (
                "Earliest date this scoring regime governs (publication of the first issue "
                "containing it, or the season start)."
            ),
            "effective_to": "Last date of the season the regime governs.",
            "evidence_source": (
                "Retained FIA regulation issues (URL and SHA-256) and article numbers."
            ),
            "publication_or_rule_date": (
                "Issue date printed in the first regulation issue in force."
            ),
        },
        "rules": rows,
        "season_rules": periods,
    }


FIELD_DEFINITIONS = {
    "season": "Championship year.",
    "round": "Jolpica calendar round, matching Gold event_id.",
    "event": "Event name as listed in the Jolpica calendar.",
    "event_id_if_known": (
        "Gold-compatible id 'season=YYYY/round=RR'. Always populated for completed calendar events."
    ),
    "in_gold": "True when this driver-event row exists in the current Gold cohort.",
    "driver": (
        "Gold driver id, mapped from the FIA name through the existing alias table. Null means "
        "unmapped (a conflict is recorded)."
    ),
    "fia_driver_name": "Driver label exactly as printed in the FIA points matrix.",
    "constructor": "Gold constructor id for the driver at this event.",
    "constructor_source": "gold_roster, formula1_com_race_result_team or jolpica_results.",
    "race_points": (
        "Grand Prix classification points excluding any fastest-lap bonus. Derived as FIA weekend"
        " points minus sprint points minus bonus_points."
    ),
    "sprint_points": (
        "Sprint points. From the FIA 'Championship Points after Sprint' document when one exists,"
        " otherwise formula1.com sprint results (2022 only). 0.0 at non-sprint events. Null when "
        "the split cannot be established."
    ),
    "bonus_points": (
        "Fastest-lap point (2022-2024 only): 1.0 when the FIA race position cell carries the F "
        "marker, the position is 1-10 and the FIA weekend total exceeds the scale value by "
        "exactly one. Otherwise 0.0."
    ),
    "penalties_or_adjustments": (
        "Classification statuses (DQ/EX), every change of this event's points value across later "
        "FIA documents, sprint revisions, and pending appeal notes printed on the latest FIA "
        "document."
    ),
    "total_event_points": (
        "Weekend championship points (race + sprint + bonus) for this event from the latest FIA "
        "championship points document carrying the event column."
    ),
    "classification_source": (
        "FIA race position cell, FIA classification registry rows for the event, and the Gold "
        "label audit reference when in Gold."
    ),
    "scoring_source": (
        "Latest FIA document supplying total_event_points, and the source of the sprint split."
    ),
    "published_at": (
        "UTC upper bound for publication of the first FIA document from which total_event_points "
        "has held its current value. Registry clocks are Paris wall time and are interpreted as "
        "UTC+1, which is the later bound when CEST applies."
    ),
    "first_published_at": (
        "UTC upper bound for the first FIA document carrying any value for this event."
    ),
    "points_timeline": (
        "The first FIA document carrying this driver's weekend value for this event and every "
        "later document that changed it, in publication order. The value known at time t is the "
        "last entry published at or before t."
    ),
    "fia_documents_carrying_value": (
        "Number of FIA documents (including later cumulative matrices) carrying this value."
    ),
    "revision_status": (
        "unrevised, revised (value changed in a later FIA document), or pending_appeal (latest "
        "FIA document prints an unresolved appeal note naming this event)."
    ),
    "evidence_quality": "See evidence_quality_levels.",
    "checks": (
        "Regulation-scale and cross-source checks. formula1.com and Jolpica are current-state "
        "sources and are never used to override FIA values."
    ),
}
EVIDENCE_QUALITY = {
    "fia_official_formula1_confirmed": (
        "Weekend total from the event's own FIA championship points document chain, sprint split "
        "from an FIA after-sprint document where applicable, and formula1.com agrees on race and "
        "sprint points."
    ),
    "fia_total_formula1_sprint_split": (
        "2022 sprint events: FIA weekend total; sprint split from formula1.com sprint results; "
        "formula1.com race points agree with the remainder."
    ),
    "fia_later_document_only_formula1_confirmed": (
        "The event's own FIA points document does not exist; the value is from the next FIA "
        "matrix that carries the event column, confirmed by formula1.com."
    ),
    "fia_official_unconfirmed": "FIA value present; no secondary confirmation available.",
    "conflict_secondary_source": (
        "FIA value present but formula1.com disagrees. FIA value is retained; see conflicts."
    ),
    "unknown_sprint_split": "Sprint split could not be audited.",
    "conflict_unmapped_driver": "FIA driver label could not be mapped to a Gold driver id.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--root", type=Path, default=Path.cwd())
    build_parser = commands.add_parser("build")
    build_parser.add_argument("collection", type=Path)
    build_parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "collect":
        print(collect(args.root))
        return
    collection = json.loads(args.collection.read_text(encoding="utf-8"))
    outputs = {
        "data/audit/scoring_rules.json": build_rules(args.root, collection),
        "data/audit/event_points_evidence.json": build(args.root, args.collection),
    }
    for relative, payload in outputs.items():
        output = args.root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        output.write_text(text, encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
