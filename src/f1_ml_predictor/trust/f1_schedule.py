"""Validate reviewed F1 schedule articles against retained HTML.

This extracts source claims. It does not independently prove an unchanged
historical webpage or certify a benchmark. Only explicitly mapped season/event
identities are supported; calendar revisions need a separate source.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

SCHEDULE_URLS = {
    2022: "https://www.formula1.com/en/latest/article/"
    "2022-f1-grand-prix-start-times-confirmed.2JejCjCeFvatOkaHsFQYC2",
    2025: "https://www.formula1.com/en/latest/article/"
    "f1-announces-race-start-times-for-2025-season.490KLLD7T1AAM7wQl28tn6",
    2026: "https://www.formula1.com/en/latest/article/"
    "official-grand-prix-start-times-for-2026-f1-season-confirmed."
    "2UgPfArqH76tzlOYh21jSG.2UgPfArqH76tzlOYh21jSG",
}
# The retained 2026 request repeats the ID; the publisher's JSON-LD does not.
ARTICLE_URLS = {
    2022: SCHEDULE_URLS[2022],
    2025: SCHEDULE_URLS[2025],
    2026: SCHEDULE_URLS[2026].rsplit(".", 1)[0],
}
_HEADLINES = {
    2022: "2022 F1 Grand Prix start times confirmed",
    2025: "F1 announces race start times for 2025 season",
    2026: "Official Grand Prix start times for 2026 F1 season confirmed",
}
_HEADERS = {
    2022: ("GRAND PRIX", "DATE", "LOCAL TIME", "UTC"),
    2025: ("RACE", "DATE", "LOCAL START TIME", "(GMT)"),
    2026: (
        "Venue, race date",
        "Sprint (local time)",
        "Qualifying (local time)",
        "Race (local time)",
    ),
}
# Administrative mappings are explicit, never inferred from row ordinal.
_VENUES = {
    "albert_park": ("Australia", "Australian Grand Prix", "Australia/Melbourne"),
    "shanghai": ("China", "Chinese Grand Prix", "Asia/Shanghai"),
    "interlagos": ("Sao Paulo", "Sao Paulo Grand Prix", "America/Sao_Paulo"),
    "vegas": ("Las Vegas", "Las Vegas Grand Prix", "America/Los_Angeles"),
    "losail": ("Qatar", "Qatar Grand Prix", "Asia/Qatar"),
    "yas_marina": ("Abu Dhabi", "Abu Dhabi Grand Prix", "Asia/Dubai"),
    "suzuka": ("Japan", "Japanese Grand Prix", "Asia/Tokyo"),
    "bahrain": ("Bahrain", "Bahrain Grand Prix", "Asia/Bahrain"),
    "jeddah": ("Saudi Arabia", "Saudi Arabian Grand Prix", "Asia/Riyadh"),
    "miami": ("Miami", "Miami Grand Prix", "America/New_York"),
    "imola": ("Emilia-Romagna", "Emilia Romagna Grand Prix", "Europe/Rome"),
    "monaco": ("Monaco", "Monaco Grand Prix", "Europe/Monaco"),
    "catalunya": ("Spain", "Spanish Grand Prix", "Europe/Madrid"),
    "villeneuve": ("Canada", "Canadian Grand Prix", "America/Toronto"),
    "red_bull_ring": ("Austria", "Austrian Grand Prix", "Europe/Vienna"),
    "silverstone": ("Great Britain", "British Grand Prix", "Europe/London"),
    "spa": ("Belgium", "Belgian Grand Prix", "Europe/Brussels"),
    "hungaroring": ("Hungary", "Hungarian Grand Prix", "Europe/Budapest"),
    "zandvoort": ("Netherlands", "Dutch Grand Prix", "Europe/Amsterdam"),
    "monza": ("Italy", "Italian Grand Prix", "Europe/Rome"),
    "baku": ("Azerbaijan", "Azerbaijan Grand Prix", "Asia/Baku"),
    "marina_bay": ("Singapore", "Singapore Grand Prix", "Asia/Singapore"),
    "americas": ("United States", "United States Grand Prix", "America/Chicago"),
    "rodriguez": ("Mexico City", "Mexico City Grand Prix", "America/Mexico_City"),
    "madring": ("Madrid", "Spanish Grand Prix", "Europe/Madrid"),
    "ricard": ("France", "French Grand Prix", "Europe/Paris"),
    "sepang": ("Malaysia", "Bahrain Grand Prix in Malaysia", "Asia/Kuala_Lumpur"),
}
_EVENT_NAMES = {
    (2022, "interlagos"): "São Paulo Grand Prix",
    (2026, "catalunya"): "Barcelona Grand Prix",
}
_VENUE_NAMES = {
    (2022, "interlagos"): "Brazil",
    (2022, "imola"): "Emilia Romagna",
    (2022, "rodriguez"): "Mexico",
    (2022, "silverstone"): "Great Britain",
    (2026, "catalunya"): "Barcelona",
    (2026, "madring"): "Spain",
}


# A relocated event is absent from its season's start-time article. Each entry
# binds one reviewed official article whose prose states the race start.
RACE_TIME_ARTICLES = {
    (2026, 16): {
        "circuit_id": "sepang",
        "url": "https://www.formula1.com/en/latest/article/"
        "what-time-is-the-formula-1-2026-bahrain-grand-prix-in-malaysia-and-how-can-i-watch-it."
        "3gFLqniY3acdKlkPjaN66d",
        "headline": "What time is the Formula 1 2026 Bahrain Grand Prix in Malaysia "
        "and how can I watch it?",
        "published_at_utc": "2026-09-29T20:56:06.110000+00:00",
    },
}


def canonical_event_name(season: int, circuit_id: str) -> str:
    """Administrative event label for an explicitly reviewed circuit identity."""
    return _EVENT_NAMES.get((season, circuit_id), _VENUES[circuit_id][1])


_TIMETABLE_HEADING_ALIASES = {
    "red_bull_ring": ("osterreich",),
    "monza": ("italia",),
    "catalunya": ("espana",),
    "rodriguez": ("mexico",),
}
_ROUNDS = {
    **{
        (2022, circuit): round_number
        for round_number, circuit in enumerate(
            (
                "bahrain",
                "jeddah",
                "albert_park",
                "imola",
                "miami",
                "catalunya",
                "monaco",
                "baku",
                "villeneuve",
                "silverstone",
                "red_bull_ring",
                "ricard",
                "hungaroring",
                "spa",
                "zandvoort",
                "monza",
                "marina_bay",
                "suzuka",
                "americas",
                "rodriguez",
                "interlagos",
                "yas_marina",
            ),
            start=1,
        )
    },
    (2025, "albert_park"): 1,
    (2025, "shanghai"): 2,
    (2025, "interlagos"): 21,
    (2025, "vegas"): 22,
    (2025, "losail"): 23,
    (2025, "yas_marina"): 24,
    (2026, "albert_park"): 1,
    (2026, "shanghai"): 2,
    (2026, "suzuka"): 3,
    (2026, "spa"): 10,
    (2026, "hungaroring"): 11,
    (2026, "zandvoort"): 12,
    (2026, "monza"): 13,
    (2026, "baku"): 15,
    (2026, "silverstone"): 9,
    (2026, "miami"): 4,
    (2026, "villeneuve"): 5,
    (2026, "monaco"): 6,
    (2026, "catalunya"): 7,
    (2026, "red_bull_ring"): 8,
    (2026, "madring"): 14,
    (2025, "suzuka"): 3,
    (2025, "bahrain"): 4,
    (2025, "jeddah"): 5,
    (2025, "miami"): 6,
    (2025, "imola"): 7,
    (2025, "monaco"): 8,
    (2025, "catalunya"): 9,
    (2025, "villeneuve"): 10,
    (2025, "red_bull_ring"): 11,
    (2025, "silverstone"): 12,
    (2025, "spa"): 13,
    (2025, "hungaroring"): 14,
    (2025, "zandvoort"): 15,
    (2025, "monza"): 16,
    (2025, "baku"): 17,
    (2025, "marina_bay"): 18,
    (2025, "americas"): 19,
    (2025, "rodriguez"): 20,
}
_MONTHS = {
    name.lower(): index
    for index, pair in enumerate(
        [
            ("January", "Jan"),
            ("February", "Feb"),
            ("March", "Mar"),
            ("April", "Apr"),
            ("May", "May"),
            ("June", "Jun"),
            ("July", "Jul"),
            ("August", "Aug"),
            ("September", "Sep"),
            ("October", "Oct"),
            ("November", "Nov"),
            ("December", "Dec"),
        ],
        start=1,
    )
    for name in pair
}


class ScheduleValidationError(ValueError):
    """Retained schedule contents do not support the requested event claim."""


@dataclass(frozen=True)
class ScheduledRace:
    publication_at: datetime
    available_by: datetime
    publication_precision_seconds: int
    race_start: datetime
    season: int
    round: int
    circuit_id: str
    event_name: str
    source_url: str
    source_row: tuple[str, ...]
    timezone_basis: str
    html_sha256: str


class _ScheduleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self.times: list[str] = []
        self.scripts: list[str] = []
        self.tables: list[list[tuple[str, ...]]] = []
        self.capture: tuple[str, list[str]] | None = None
        self.table: list[tuple[str, ...]] | None = None
        self.row: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self.table is not None:
                raise ScheduleValidationError("nested schedule tables are unsupported")
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        if (
            tag in {"h1", "time"}
            or (tag == "script" and dict(attrs).get("type") == "application/ld+json")
            or (tag in {"td", "th"} and self.row is not None)
        ):
            if self.capture is not None:
                raise ScheduleValidationError("nested schedule metadata is unsupported")
            self.capture = (tag, [])

    def handle_data(self, data: str) -> None:
        if self.capture is not None:
            self.capture[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.capture is not None and tag == self.capture[0]:
            value = "".join(self.capture[1])
            if tag == "script":
                self.scripts.append(value)
            elif tag == "h1":
                self.headings.append(_space(value))
            elif tag == "time":
                self.times.append(_space(value))
            elif self.row is not None:
                self.row.append(_space(value))
            self.capture = None
        if tag == "tr" and self.row is not None and self.table is not None:
            self.table.append(tuple(self.row))
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def _space(value: str) -> str:
    return " ".join(value.split())


def _alias(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", _space(value))
        if not unicodedata.combining(char)
    ).casefold()


def _identity_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _alias(value)))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ScheduleValidationError("schedule claims require timezone-aware timestamps")
    return value.astimezone(UTC)


def _stamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ScheduleValidationError("article publication timestamp is missing")
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ScheduleValidationError("invalid article publication timestamp") from exc


def _entities(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [entity for item in value for entity in _entities(item)]
    if isinstance(value, dict):
        return [value, *_entities(value.get("@graph", []))]
    return []


def _visible_time(value: str) -> datetime | None:
    match = re.fullmatch(r"([A-Za-z]{3}) (\d{1,2}), (\d{4}) (\d{1,2}):(\d{2})(am|pm) UTC", value)
    if match is None:
        return None
    month, day, year, hour, minute, period = match.groups()
    hour_number = int(hour)
    if not 1 <= hour_number <= 12 or month.lower() not in _MONTHS:
        raise ScheduleValidationError("invalid visible UTC publication time")
    return datetime(
        int(year),
        _MONTHS[month.lower()],
        int(day),
        hour_number % 12 + (12 if period == "pm" else 0),
        int(minute),
        tzinfo=UTC,
    )


def _date(value: str, season: int) -> datetime:
    match = re.fullmatch(r"([A-Za-z]+) (\d{1,2})", value)
    if match is None or match[1].lower() not in _MONTHS:
        raise ScheduleValidationError("unsupported race date in schedule row")
    return datetime(season, _MONTHS[match[1].lower()], int(match[2]))


def _clock(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{2}):?(\d{2})", value)
    if match is None or int(match[1]) > 23 or int(match[2]) > 59:
        raise ScheduleValidationError("invalid race start clock")
    return int(match[1]), int(match[2])


def _local_utc(local: datetime, zone_name: str) -> datetime:
    zone = ZoneInfo(zone_name)
    possibilities = {
        candidate
        for fold in (0, 1)
        if (candidate := local.replace(tzinfo=zone, fold=fold).astimezone(UTC))
        .astimezone(zone)
        .replace(tzinfo=None)
        == local
    }
    if len(possibilities) != 1:
        raise ScheduleValidationError("ambiguous or nonexistent local race time")
    return possibilities.pop()


def validate_f1_schedule(
    html: str,
    *,
    season: int,
    event_name: str,
    circuit_id: str,
    claimed_publication: datetime,
    claimed_race_start: datetime,
    source_url: str | None = None,
    round_number: int | None = None,
    prediction_timestamp: datetime | None = None,
) -> ScheduledRace:
    """Bind an event claim to article metadata and one exact schedule row.

    Rounded publication claims may match the visible minute, but the returned
    publication uses JSON-LD seconds and availability conservatively ends that
    minute. Any article modification must also be known by the supplied cutoff.
    The caller retains and hashes the original response bytes separately.
    """
    if (season, circuit_id) not in _ROUNDS:
        raise ScheduleValidationError("event is outside the reviewed winter schedule cohort")
    expected_url = SCHEDULE_URLS[season]
    canonical_url = ARTICLE_URLS[season]
    if source_url is not None and source_url not in {expected_url, canonical_url}:
        raise ScheduleValidationError("schedule source URL is not the reviewed official article")
    venue, expected_event, zone_name = _VENUES[circuit_id]
    if _alias(event_name) != _alias(_EVENT_NAMES.get((season, circuit_id), expected_event)):
        raise ScheduleValidationError("event name does not match the reviewed circuit")
    expected_round = _ROUNDS[season, circuit_id]
    if round_number is not None and round_number != expected_round:
        raise ScheduleValidationError("round does not match the reviewed circuit and season")
    parser = _ScheduleHTML()
    parser.feed(html)
    parser.close()
    if parser.capture is not None or parser.table is not None:
        raise ScheduleValidationError("incomplete schedule HTML")
    if parser.headings != [_HEADLINES[season]]:
        raise ScheduleValidationError("article heading does not identify the requested season")
    articles: list[dict[str, Any]] = []
    for script in parser.scripts:
        try:
            records = _entities(json.loads(script))
        except json.JSONDecodeError as exc:
            raise ScheduleValidationError("invalid article JSON-LD") from exc
        articles.extend(
            item
            for item in records
            if item.get("@type") == "NewsArticle" and item.get("url") == canonical_url
        )
    if len(articles) != 1:
        raise ScheduleValidationError("one matching official NewsArticle entity is required")
    article = articles[0]
    if any(article.get(key, canonical_url) != canonical_url for key in ("@id", "mainEntityOfPage")):
        raise ScheduleValidationError("article canonical identity is inconsistent")
    headline = article.get("headline", "")
    if not isinstance(headline, str) or not (
        headline == _HEADLINES[season] or headline.startswith(_HEADLINES[season] + " | Formula 1")
    ):
        raise ScheduleValidationError("article JSON-LD headline contradicts the heading")
    published = _stamp(article.get("datePublished"))
    modified = _stamp(article.get("dateModified", article.get("datePublished")))
    if modified < published and season != 2022:
        raise ScheduleValidationError("article modification precedes publication")
    visible = [stamp for value in parser.times if (stamp := _visible_time(value)) is not None]
    minute = published.replace(second=0, microsecond=0)
    if season == 2022:
        # This article displays its publication day at midnight while JSON-LD
        # supplies the later exact time. Use the later time as availability.
        if len(visible) != 1 or visible[0].date() != published.date() or visible[0] > published:
            raise ScheduleValidationError("visible UTC stamp contradicts article publication")
    elif visible != [minute]:
        raise ScheduleValidationError("visible UTC stamp contradicts article publication")
    claimed = _utc(claimed_publication)
    if claimed not in {published, minute}:
        raise ScheduleValidationError("claimed schedule publication contradicts source contents")
    available_by = max(published, modified).replace(second=0, microsecond=0) + timedelta(minutes=1)
    if modified != published and prediction_timestamp is None:
        raise ScheduleValidationError("modified article needs a reviewed historical cutoff")
    if prediction_timestamp is not None and available_by > _utc(prediction_timestamp):
        raise ScheduleValidationError("schedule article contents were not known by cutoff")
    tables = [table for table in parser.tables if table and table[0] == _HEADERS[season]]
    if len(tables) != 1:
        raise ScheduleValidationError("one reviewed season schedule table is required")
    rows = tables[0][1:]
    if any(len(row) != 4 for row in rows):
        raise ScheduleValidationError("schedule row schema is inconsistent")
    matches = [
        row
        for row in rows
        if _alias(row[0].rstrip(" *") if season in {2022, 2025} else row[0].partition(",")[0])
        == _alias(_VENUE_NAMES.get((season, circuit_id), venue))
    ]
    if len(matches) != 1:
        raise ScheduleValidationError("event schedule row is missing or duplicated")
    row = matches[0]
    date_text = row[1] if season in {2022, 2025} else row[0].partition(",")[2].strip()
    local = _date(date_text, season)
    hour, minute_number = _clock(row[2] if season in {2022, 2025} else row[3])
    race_start = _local_utc(local.replace(hour=hour, minute=minute_number), zone_name)
    if season in {2022, 2025} and _clock(row[3]) != (race_start.hour, race_start.minute):
        raise ScheduleValidationError("local start time contradicts the printed GMT clock")
    if race_start != _utc(claimed_race_start):
        raise ScheduleValidationError("claimed UTC race start contradicts the schedule row")
    return ScheduledRace(
        published,
        available_by,
        60,
        race_start,
        season,
        expected_round,
        circuit_id,
        event_name,
        source_url or expected_url,
        row,
        zone_name,
        hashlib.sha256(html.encode("utf-8")).hexdigest(),
    )


def validate_event_timetable(
    html: str,
    *,
    season: int,
    round_number: int,
    event_name: str,
    circuit_id: str,
    claimed_publication: datetime,
    claimed_race_start: datetime | None,
    source_url: str,
    prediction_timestamp: datetime,
) -> ScheduledRace:
    """Bind one published F1 event timetable to its exact race start row."""
    parsed_url = urlsplit(source_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != "www.formula1.com"
        or parsed_url.port is not None
        or not re.fullmatch(
            rf"/en/latest/article/[a-z0-9-]+-{season}-timetable\.[A-Za-z0-9]+",
            parsed_url.path,
        )
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ScheduleValidationError("event timetable URL is not an official season article")
    if season not in {2023, 2024} or circuit_id not in _VENUES:
        raise ScheduleValidationError("unsupported historical event timetable")
    if _utc(claimed_publication) >= _utc(prediction_timestamp):
        raise ScheduleValidationError("event timetable was published after cutoff")
    parser = _ScheduleHTML()
    parser.feed(html)
    parser.close()
    if parser.capture is not None or parser.table is not None or len(parser.headings) != 1:
        raise ScheduleValidationError("event timetable HTML is incomplete or ambiguous")
    heading = _alias(parser.headings[0])
    if str(season) not in heading.split() or not any(
        marker in heading
        for marker in ("grand prix", "gran premio", "grande premio", "grosser preis")
    ):
        raise ScheduleValidationError("event timetable heading contradicts season or document type")
    heading_words = _identity_words(parser.headings[0])
    labels = (
        event_name.removesuffix(" Grand Prix"),
        _VENUES[circuit_id][0],
        *_TIMETABLE_HEADING_ALIASES.get(circuit_id, ()),
    )
    if not any(
        re.search(rf"\b{re.escape(_identity_words(label))}\b", heading_words) for label in labels
    ):
        raise ScheduleValidationError("event timetable heading contradicts event identity")
    articles: list[dict[str, Any]] = []
    for script in parser.scripts:
        try:
            articles.extend(
                item
                for item in _entities(json.loads(script))
                if item.get("@type") == "NewsArticle" and item.get("url") == source_url
            )
        except json.JSONDecodeError as exc:
            raise ScheduleValidationError("invalid timetable JSON-LD") from exc
    if len(articles) != 1:
        raise ScheduleValidationError("one exact event timetable NewsArticle is required")
    article = articles[0]
    headline = _alias(article.get("headline", ""))
    if article.get("@id", source_url) != source_url or not (
        headline == heading or headline.startswith(heading + " - full timetable")
    ):
        raise ScheduleValidationError("event timetable canonical identity contradicts heading")
    published = _stamp(article.get("datePublished"))
    modified = _stamp(article.get("dateModified", article.get("datePublished")))
    if published != _utc(claimed_publication):
        raise ScheduleValidationError("timetable publication claim contradicts source")
    available_by = max(published, modified).replace(second=0, microsecond=0) + timedelta(minutes=1)
    if available_by > _utc(prediction_timestamp):
        raise ScheduleValidationError("event timetable contents were not known by cutoff")
    if _alias(event_name) != _alias(_EVENT_NAMES.get((season, circuit_id), _VENUES[circuit_id][1])):
        raise ScheduleValidationError("event name contradicts reviewed circuit")
    race_rows = [
        (table[0], row)
        for table in parser.tables
        if table and len(table[0]) == 3
        for row in table[1:]
        if len(row) == 3
        and row[0].upper() == "FORMULA 1"
        and row[1].upper().startswith("GRAND PRIX")
    ]
    if len(race_rows) != 1:
        raise ScheduleValidationError("one Formula 1 Grand Prix row is required")
    header, row = race_rows[0]
    day = re.fullmatch(r"[A-Z]+\s+(\d{1,2})(?:st|nd|rd|th)\s+([A-Z]+)", header[0])
    start = re.fullmatch(r"(\d{2}):(\d{2})\s*-\s*\d{2}:\d{2}", row[2])
    if day is None or start is None or day[2].lower() not in _MONTHS:
        raise ScheduleValidationError("unsupported event timetable race date or time")
    local = datetime(season, _MONTHS[day[2].lower()], int(day[1]), int(start[1]), int(start[2]))
    race_start = _local_utc(local, _VENUES[circuit_id][2])
    if claimed_race_start is not None and race_start != _utc(claimed_race_start):
        raise ScheduleValidationError("timetable race start contradicts claimed UTC event")
    return ScheduledRace(
        published,
        available_by,
        60,
        race_start,
        season,
        round_number,
        circuit_id,
        event_name,
        source_url,
        (header[0], *row),
        _VENUES[circuit_id][2],
        hashlib.sha256(html.encode("utf-8")).hexdigest(),
    )


def validate_fia_timetable_amendment(
    text: str,
    *,
    season: int,
    round_number: int,
    event_name: str,
    circuit_id: str,
    claimed_publication: datetime,
    claimed_race_start: datetime,
    source_url: str,
    prediction_timestamp: datetime,
    document_sha256: str,
) -> ScheduledRace:
    """Bind an approved FIA timetable revision published before prediction."""
    parsed_url = urlsplit(source_url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in {"fia.com", "www.fia.com"}:
        raise ScheduleValidationError("timetable amendment needs an official FIA PDF")
    if _utc(claimed_publication).replace(second=0, microsecond=0) + timedelta(minutes=1) > _utc(
        prediction_timestamp
    ):
        raise ScheduleValidationError("timetable amendment was not available at cutoff")
    plain = _identity_words(text)
    if (
        str(season) not in plain.split()
        or _identity_words(_VENUES[circuit_id][0]) not in plain
        or not re.search(r"approve version \d+ of the timetable", plain)
    ):
        raise ScheduleValidationError("FIA timetable amendment identity or approval is missing")
    sections = re.split(
        r"(?m)^\s*(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\s+([0-3]?\d)\s+([A-Z]+)\s+(20\d{2})\b",
        text,
    )
    matches = []
    for index in range(1, len(sections), 5):
        day_name, day, month, year, body = sections[index : index + 5]
        if int(year) != season or month.lower() not in _MONTHS:
            continue
        for line in body.splitlines():
            match = re.match(
                r"^\s*(\d{2}:\d{2})\*?\s+\d{2}:\d{2}\S*\s+FORMULA 1\s+TRACK\s+GRAND PRIX\b",
                " ".join(line.split()),
                re.I,
            )
            if match:
                hour, minute = map(int, match[1].split(":"))
                local = datetime(season, _MONTHS[month.lower()], int(day), hour, minute)
                matches.append((local, line.strip()))
    if len(matches) != 1:
        raise ScheduleValidationError("one amended FIA Grand Prix start row is required")
    race_start = _local_utc(matches[0][0], _VENUES[circuit_id][2])
    if race_start != _utc(claimed_race_start):
        raise ScheduleValidationError("FIA amendment contradicts the claimed race start")
    return ScheduledRace(
        _utc(claimed_publication),
        _utc(claimed_publication).replace(second=0, microsecond=0) + timedelta(minutes=1),
        60,
        race_start,
        season,
        round_number,
        circuit_id,
        event_name,
        source_url,
        (str(matches[0][0]), matches[0][1]),
        _VENUES[circuit_id][2],
        document_sha256,
    )


def validate_race_time_article(
    html: str,
    *,
    season: int,
    round_number: int,
    event_name: str,
    circuit_id: str,
    claimed_publication: datetime,
    claimed_race_start: datetime,
    source_url: str,
    prediction_timestamp: datetime,
) -> ScheduledRace:
    """Bind a relocated race start to one reviewed official article sentence.

    Only explicitly reviewed articles are accepted. The article must be unmodified
    since publication, published before cutoff, and state exactly one Grand Prix
    start on the race day in circuit local time.
    """
    reviewed = RACE_TIME_ARTICLES.get((season, round_number))
    if reviewed is None or reviewed["circuit_id"] != circuit_id:
        raise ScheduleValidationError("race time article is not reviewed for this event")
    if source_url != reviewed["url"]:
        raise ScheduleValidationError("race time article URL is not the reviewed article")
    if _alias(event_name) != _alias(_EVENT_NAMES.get((season, circuit_id), _VENUES[circuit_id][1])):
        raise ScheduleValidationError("event name contradicts reviewed circuit")
    parser = _ScheduleHTML()
    parser.feed(html)
    parser.close()
    if parser.capture is not None or parser.table is not None:
        raise ScheduleValidationError("incomplete race time article HTML")
    if [_alias(heading) for heading in parser.headings] != [_alias(reviewed["headline"])]:
        raise ScheduleValidationError("race time article heading contradicts review")
    articles: list[dict[str, Any]] = []
    for script in parser.scripts:
        try:
            articles.extend(
                item
                for item in _entities(json.loads(script))
                if item.get("@type") == "NewsArticle" and item.get("url") == source_url
            )
        except json.JSONDecodeError as exc:
            raise ScheduleValidationError("invalid race time article JSON-LD") from exc
    if len(articles) != 1:
        raise ScheduleValidationError("one exact race time NewsArticle is required")
    article = articles[0]
    if article.get("@id", source_url) != source_url or _alias(article.get("headline", "")) not in {
        _alias(reviewed["headline"]),
        _alias(reviewed["headline"] + " | Formula 1"),
    }:
        raise ScheduleValidationError("race time article canonical identity contradicts heading")
    published = _stamp(article.get("datePublished"))
    modified = _stamp(article.get("dateModified", article.get("datePublished")))
    if modified != published:
        raise ScheduleValidationError("race time article was modified after publication")
    if published != _utc(claimed_publication) or published != _stamp(reviewed["published_at_utc"]):
        raise ScheduleValidationError("race time article publication contradicts review")
    minute = published.replace(second=0, microsecond=0)
    visible = [stamp for value in parser.times if (stamp := _visible_time(value)) is not None]
    if visible != [minute]:
        raise ScheduleValidationError("visible UTC stamp contradicts article publication")
    available_by = minute + timedelta(minutes=1)
    if available_by > _utc(prediction_timestamp):
        raise ScheduleValidationError("race time article was not known by cutoff")
    body = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    text = _space(unescape(re.sub(r"<[^>]+>", " ", body)))
    sentences = re.findall(
        r"Grand Prix itself gets underway at (\d{2}):?(\d{2}) on "
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), ([A-Z][a-z]+) (\d{1,2})\b",
        text,
    )
    if len(sentences) != 1:
        raise ScheduleValidationError("one Grand Prix start sentence is required")
    hour, minute_number, weekday, month, day = sentences[0]
    if month.lower() not in _MONTHS or int(hour) > 23 or int(minute_number) > 59:
        raise ScheduleValidationError("unsupported race time article date or time")
    local = datetime(season, _MONTHS[month.lower()], int(day), int(hour), int(minute_number))
    if local.strftime("%A") != weekday:
        raise ScheduleValidationError("race time article weekday contradicts its date")
    zone_name = _VENUES[circuit_id][2]
    race_start = _local_utc(local, zone_name)
    if race_start != _utc(claimed_race_start):
        raise ScheduleValidationError("race time article contradicts the claimed race start")
    return ScheduledRace(
        published,
        available_by,
        60,
        race_start,
        season,
        round_number,
        circuit_id,
        event_name,
        source_url,
        (weekday, f"{month} {day}", f"{hour}:{minute_number}"),
        zone_name,
        hashlib.sha256(html.encode("utf-8")).hexdigest(),
    )
