"""Bounded FIA layout-text parsers without publication or audit inference.

Names resolve only through caller-provided exact aliases. Text parsing does not
certify a document's event, version, publication time, or suitability for Gold.
PDF extraction and those evidence checks belong to the calling audit pipeline.
"""

import re
from dataclasses import dataclass
from statistics import median
from typing import Any

import pyarrow as pa

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.trust.outcomes import DnfCategory, audited_dnf

QUALIFYING_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string(), nullable=False),
        pa.field("position", pa.int64()),
        pa.field("q1_seconds", pa.float64()),
        pa.field("q2_seconds", pa.float64()),
        pa.field("q3_seconds", pa.float64()),
    ]
)
ROSTER_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string(), nullable=False),
        pa.field("grid_position", pa.int64()),
        pa.field("start_type", pa.string(), nullable=False),
        pa.field("grid_status", pa.string(), nullable=False),
    ]
)
FINAL_INTERMEDIATE_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string(), nullable=False),
        pa.field("position", pa.int64()),
        pa.field("classified", pa.bool_(), nullable=False),
        pa.field("raw_status", pa.string(), nullable=False),
        pa.field("dnf_category", pa.string(), nullable=False),
        pa.field("dnf", pa.bool_()),
    ]
)

_MAX_TEXT = 2_000_000
_MAX_ROWS = 60
_PREFIX = re.compile(r"^\s*(\d+)(?:\s+(\d+))?\s+")
_LAP = re.compile(r"(?<![\d:])(\d{1,2}):([0-5]\d)\.(\d{3})(?![\d:])")
_FINAL_HEADER = re.compile(r"LAPS.*TIME.*(?:GAP|INT|KM/H)", re.IGNORECASE)
_NATIONS = frozenset(
    {
        "NLD",
        "NED",
        "MEX",
        "GBR",
        "MCO",
        "MON",
        "ESP",
        "AUS",
        "CAN",
        "ARG",
        "FRA",
        "NZL",
        "JPN",
        "JAP",
        "THA",
        "FIN",
        "CHN",
        "DNK",
        "DEN",
        "DEU",
        "GER",
        "BRA",
        "BEL",
        "ITA",
        "USA",
        "POR",
        "POL",
        "RUS",
        "IND",
        "IDN",
        "SUI",
        "AUT",
        "ZAF",
        "SWE",
    }
)
_STATUS = re.compile(
    r"\b(?:DSQ|DQ|DISQUALIFIED|DNS|DID\s+NOT\s+START|FINISHED|"
    r"RETIRED(?:\s+(?:MECHANICAL|INCIDENT|OTHER))?|DNF)\b",
    re.IGNORECASE,
)


class ParsingFailure(ValueError):
    """The supported layout or exact identity mapping cannot establish a table."""


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _aliases(values: dict[str, str], kind: EntityKind) -> dict[str, str]:
    if not isinstance(values, dict) or not values:
        raise ParsingFailure(f"explicit {kind.value} aliases are required")
    normalized: dict[str, str] = {}
    for name, identifier in values.items():
        if not isinstance(name, str) or not _normalize(name):
            raise ParsingFailure("alias display names must be nonempty strings")
        EntityId(kind, identifier)
        alias = _normalize(name)
        if alias in normalized and normalized[alias] != identifier:
            raise ParsingFailure("conflicting exact display-name aliases")
        normalized[alias] = identifier
    return dict(sorted(normalized.items(), key=lambda item: -len(item[0])))


def _lookup_prefix(value: str, aliases: dict[str, str], kind: str) -> tuple[str, str]:
    for alias, identifier in aliases.items():
        if value == alias:
            return identifier, ""
        if value.startswith(alias + " "):
            return identifier, value[len(alias) :].strip()
        if value.startswith(alias + "* "):
            return identifier, value[len(alias) + 1 :].strip()
    raise ParsingFailure(f"unmapped exact {kind} name in numeric table row: {value[:100]}")


@dataclass(frozen=True)
class _Row:
    text: str
    position: int | None
    driver: str
    constructor: str
    tail: str
    classified: bool
    section_status: str | None = None


def _identity(
    body: str, drivers: dict[str, str], constructors: dict[str, str], *, roster: bool = False
) -> tuple[str, str, str]:
    driver, remainder = _lookup_prefix(_normalize(body), drivers, "driver")
    remainder = remainder.lstrip("* ")
    # The nationality column is often blank in timing sheets, but populated in
    # entry lists. It is not used to guess or normalize a driver identity.
    nation = re.match(r"^[A-Z]{3}\s+", remainder)
    if nation is not None and nation[0].strip() in _NATIONS:
        remainder = remainder[nation.end() :]
    constructor, tail = _lookup_prefix(remainder, constructors, "constructor")
    if not roster and tail and not re.match(r"^(?:\d|[-])", tail) and not _STATUS.match(tail):
        raise ParsingFailure("constructor alias does not match the complete timing-sheet entrant")
    if roster:
        # An entry list has both Team and Constructor columns. When both names
        # are explicitly mapped, they must agree. No car number becomes a grid.
        exact_cells = [
            constructors[cell]
            for raw_cell in re.split(r"\s{2,}", body.strip())
            if (cell := _normalize(raw_cell)) in constructors
        ]
        if not tail or tail not in constructors:
            raise ParsingFailure(
                "entry-list constructor alias does not match a complete name column"
            )
        if any(value != constructor for value in exact_cells) or (
            tail in constructors and constructors[tail] != constructor
        ):
            raise ParsingFailure("entry-list Team and Constructor identities disagree")
    return driver, constructor, tail


def _lines(text: str, event: EventId) -> list[str]:
    if not isinstance(event, EventId):
        raise ParsingFailure("a canonical EventId is required")
    if not isinstance(text, str) or not text.strip() or len(text) > _MAX_TEXT:
        raise ParsingFailure("nonempty bounded extracted PDF text is required")
    return text.splitlines()


def _timing_rows(
    text: str,
    event: EventId,
    driver_aliases: dict[str, str],
    constructor_aliases: dict[str, str],
    *,
    qualifying: bool,
) -> list[_Row]:
    drivers = _aliases(driver_aliases, EntityKind.DRIVER)
    constructors = _aliases(constructor_aliases, EntityKind.CONSTRUCTOR)
    active = False
    classified = True
    section_status: str | None = None
    rows: list[_Row] = []
    for line in _lines(text, event):
        normalized = _normalize(line).upper()
        header = "DRIVER" in normalized and (
            all(name in normalized for name in ("Q1", "Q2", "Q3"))
            if qualifying
            else _FINAL_HEADER.search(normalized) is not None
        )
        if header:
            active = True
            continue
        if not active:
            continue
        if (
            any(
                marker in normalized
                for marker in (
                    "POLE POSITION",
                    "FASTEST LAP",
                    "TIMEKEEPER",
                    "* PENALTIES",
                )
            )
            or normalized == "NOTES"
            or "KM/H" in normalized
        ):
            break
        if "NOT CLASSIFIED" in normalized or normalized in {"DISQUALIFIED", "DID NOT START"}:
            classified = False
            section_status = normalized if normalized in {"DISQUALIFIED", "DID NOT START"} else None
            continue
        prefix = _PREFIX.match(line)
        if prefix is None:
            continue
        if classified and prefix[2] is None:
            raise ParsingFailure(
                "classified timing row requires both position and car-number columns"
            )
        if not classified and prefix[2] is not None:
            raise ParsingFailure("unsupported ordinal in a not-classified timing section")
        position = int(prefix[1]) if classified else None
        if position is not None and not 1 <= position <= _MAX_ROWS:
            raise ParsingFailure("timing position is outside the bounded field")
        driver, constructor, tail = _identity(line[prefix.end() :], drivers, constructors)
        rows.append(_Row(line, position, driver, constructor, tail, classified, section_status))
        if len(rows) > _MAX_ROWS:
            raise ParsingFailure("timing table exceeds the bounded field")
    if not active:
        raise ParsingFailure("supported qualifying/race table header was not found")
    _unique(rows)
    return rows


def _unique(rows: list[_Row]) -> None:
    if not rows:
        raise ParsingFailure("supported table contains no driver rows")
    if len({row.driver for row in rows}) != len(rows):
        raise ParsingFailure("duplicate canonical driver in extracted table")
    positions = [row.position for row in rows if row.position is not None]
    if len(set(positions)) != len(positions):
        raise ParsingFailure("duplicate classification position in extracted table")


def parse_qualifying_text(
    text: str,
    event: EventId,
    driver_aliases: dict[str, str],
    constructor_aliases: dict[str, str],
) -> pa.Table:
    """Parse Q1/Q2/Q3 layout columns, preserving missing stages and ordinals.

    Wall clocks with two colons and Q1 percentage values cannot become lap times.
    Complete three-stage rows establish the column anchors for sparse rows.
    Compact rows with only consecutive stages are also supported. Unsupported
    column drift or ambiguity fails instead of shifting a Q3 time into Q2.
    """
    rows = _timing_rows(text, event, driver_aliases, constructor_aliases, qualifying=True)
    matches = [list(_LAP.finditer(row.text)) for row in rows]
    if any(len(values) > 3 for values in matches):
        raise ParsingFailure("qualifying row has more than three session lap times")
    complete = [values for values in matches if len(values) == 3]
    anchors = (
        [median(values[stage].start() for values in complete) for stage in range(3)]
        if complete
        else []
    )
    output: list[dict[str, Any]] = []
    for row, values in zip(rows, matches, strict=True):
        times: list[float | None] = [None, None, None]
        for index, match in enumerate(values):
            if anchors:
                stage = min(range(3), key=lambda item: abs(anchors[item] - match.start()))
                if abs(anchors[stage] - match.start()) > 12 or times[stage] is not None:
                    raise ParsingFailure("ambiguous qualifying session-column alignment")
            else:
                if _STATUS.search(row.tail) and values:
                    raise ParsingFailure(
                        "sparse qualifying status/time layout needs session-column anchors"
                    )
                stage = index
            seconds = int(match[1]) * 60 + int(match[2]) + int(match[3]) / 1000
            if seconds <= 0:
                raise ParsingFailure("qualifying lap times must be positive")
            times[stage] = seconds
        output.append(
            {
                "event_id": event.partition(),
                "driver_id": row.driver,
                "constructor_id": row.constructor,
                "position": row.position,
                "q1_seconds": times[0],
                "q2_seconds": times[1],
                "q3_seconds": times[2],
            }
        )
    return pa.Table.from_pylist(output, schema=QUALIFYING_SCHEMA)


def parse_roster_text(
    text: str,
    event: EventId,
    driver_aliases: dict[str, str],
    constructor_aliases: dict[str, str],
) -> pa.Table:
    """Parse the full event entry table, excluding a subsequent P1-only list.

    Car numbers can have appended superscript footnotes in extracted text. They
    are ignored, and never treated as starting positions or inferred identities.
    """
    drivers = _aliases(driver_aliases, EntityKind.DRIVER)
    constructors = _aliases(constructor_aliases, EntityKind.CONSTRUCTOR)
    active = False
    has_tla = False
    rows: list[_Row] = []
    for line in _lines(text, event):
        normalized = _normalize(line).upper()
        if all(name in normalized for name in ("DRIVER", "TEAM", "CONSTRUCTOR")):
            active = True
            has_tla = "TLA" in normalized.split()
            continue
        if not active:
            continue
        if any(
            marker in normalized
            for marker in (
                "IN ADDITION TO",
                "PART IN P1",
                "P1 ONLY",
                "RESERVE DRIVERS",
                "THE STEWARDS",
            )
        ):
            break
        prefix = _PREFIX.match(line)
        if prefix is None:
            continue
        if prefix[2] is not None:
            raise ParsingFailure("entry list requires one car-number column, not a grid ordinal")
        body = line[prefix.end() :]
        if has_tla:
            tla = re.match(r"\s*[A-Z]{3}\s+", body)
            if tla is None:
                raise ParsingFailure("entry-list TLA column does not match the declared layout")
            body = body[tla.end() :]
        driver, constructor, tail = _identity(body, drivers, constructors, roster=True)
        rows.append(_Row(line, None, driver, constructor, tail, False))
        if len(rows) > _MAX_ROWS:
            raise ParsingFailure("entry table exceeds the bounded field")
    if not active:
        raise ParsingFailure("supported entry-list Driver/Team/Constructor header was not found")
    _unique(rows)
    return pa.Table.from_pylist(
        [
            {
                "event_id": event.partition(),
                "driver_id": row.driver,
                "constructor_id": row.constructor,
                "grid_position": None,
                "start_type": "unknown",
                "grid_status": "unknown",
            }
            for row in rows
        ],
        schema=ROSTER_SCHEMA,
    )


def _outcome_status(row: _Row) -> tuple[str, DnfCategory]:
    match = _STATUS.search(row.tail)
    if match is None and row.section_status is None:
        return "classification_only" if row.classified else "not_classified", DnfCategory.UNKNOWN
    raw = _normalize(match[0] if match is not None else str(row.section_status))
    token = raw.upper()
    categories = {
        "FINISHED": DnfCategory.FINISHED,
        "DNS": DnfCategory.DID_NOT_START,
        "DID NOT START": DnfCategory.DID_NOT_START,
        "DSQ": DnfCategory.DISQUALIFIED,
        "DQ": DnfCategory.DISQUALIFIED,
        "DISQUALIFIED": DnfCategory.DISQUALIFIED,
        "RETIRED MECHANICAL": DnfCategory.RETIRED_MECHANICAL,
        "RETIRED INCIDENT": DnfCategory.RETIRED_INCIDENT,
        "RETIRED OTHER": DnfCategory.RETIRED_OTHER,
    }
    category = categories.get(token, DnfCategory.UNKNOWN)
    if row.classified and category in {DnfCategory.DID_NOT_START, DnfCategory.DISQUALIFIED}:
        raise ParsingFailure("DNS/DSQ status contradicts the classified section")
    return raw, category


def parse_final_text(
    text: str,
    event: EventId,
    driver_aliases: dict[str, str],
    constructor_aliases: dict[str, str],
) -> pa.Table:
    """Parse classification only; no final audit flag or label clock is inferred.

    A classified retirement keeps its printed position. An unclassified car has
    no invented position. Generic DNF supplies no audited retirement reason, so
    it remains UNKNOWN with DNF missing. Laps, elapsed times, gaps and ordinal
    classification alone never establish a finished/retired DNF target.
    """
    rows = _timing_rows(text, event, driver_aliases, constructor_aliases, qualifying=False)
    output = []
    for row in rows:
        raw, category = _outcome_status(row)
        output.append(
            {
                "event_id": event.partition(),
                "driver_id": row.driver,
                "constructor_id": row.constructor,
                "position": row.position,
                "classified": row.classified,
                "raw_status": raw,
                "dnf_category": category.value,
                "dnf": audited_dnf(category),
            }
        )
    return pa.Table.from_pylist(output, schema=FINAL_INTERMEDIATE_SCHEMA)
