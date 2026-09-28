"""Validate the two reviewed F1 schedule articles against retained HTML.

This extracts source claims. It does not independently prove an unchanged
historical webpage or certify a benchmark. Only the reviewed winter cohort is
supported; current calendar arbitration requires a separate revision source.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from zoneinfo import ZoneInfo

SCHEDULE_URLS = {
    2025: "https://www.formula1.com/en/latest/article/"
    "f1-announces-race-start-times-for-2025-season.490KLLD7T1AAM7wQl28tn6",
    2026: "https://www.formula1.com/en/latest/article/"
    "official-grand-prix-start-times-for-2026-f1-season-confirmed."
    "2UgPfArqH76tzlOYh21jSG.2UgPfArqH76tzlOYh21jSG",
}
# The retained 2026 request repeats the ID; the publisher's JSON-LD does not.
ARTICLE_URLS = {2025: SCHEDULE_URLS[2025], 2026: SCHEDULE_URLS[2026].rsplit(".", 1)[0]}
_HEADLINES = {
    2025: "F1 announces race start times for 2025 season",
    2026: "Official Grand Prix start times for 2026 F1 season confirmed",
}
_HEADERS = {
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
}
_ROUNDS = {
    (2025, "albert_park"): 1,
    (2025, "shanghai"): 2,
    (2025, "interlagos"): 21,
    (2025, "vegas"): 22,
    (2025, "losail"): 23,
    (2025, "yas_marina"): 24,
    (2026, "albert_park"): 1,
    (2026, "shanghai"): 2,
    (2026, "suzuka"): 3,
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
    if _alias(event_name) != _alias(expected_event):
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
    if modified < published:
        raise ScheduleValidationError("article modification precedes publication")
    visible = [stamp for value in parser.times if (stamp := _visible_time(value)) is not None]
    minute = published.replace(second=0, microsecond=0)
    if visible != [minute]:
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
        if _alias(row[0] if season == 2025 else row[0].partition(",")[0]) == _alias(venue)
    ]
    if len(matches) != 1:
        raise ScheduleValidationError("event schedule row is missing or duplicated")
    row = matches[0]
    date_text = row[1] if season == 2025 else row[0].partition(",")[2].strip()
    local = _date(date_text, season)
    hour, minute_number = _clock(row[2] if season == 2025 else row[3])
    race_start = _local_utc(local.replace(hour=hour, minute=minute_number), zone_name)
    if season == 2025 and _clock(row[3]) != (race_start.hour, race_start.minute):
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
