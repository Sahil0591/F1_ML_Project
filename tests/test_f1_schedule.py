import json
from datetime import UTC, datetime

import pytest

from f1_ml_predictor.trust.f1_schedule import (
    ARTICLE_URLS,
    RACE_TIME_ARTICLES,
    SCHEDULE_URLS,
    ScheduleValidationError,
    validate_event_timetable,
    validate_f1_schedule,
    validate_fia_timetable_amendment,
    validate_race_time_article,
)

_HEADLINES = {
    2025: "F1 announces race start times for 2025 season",
    2026: "Official Grand Prix start times for 2026 F1 season confirmed",
}
_HEADERS = {
    2025: ["RACE", "DATE", "LOCAL START TIME", "(GMT)"],
    2026: [
        "Venue, race date",
        "Sprint (local time)",
        "Qualifying (local time)",
        "Race (local time)",
    ],
}
_STAMP = {2025: "2025-02-03T17:02:27.315Z", 2026: "2025-09-16T09:07:24.791Z"}
_VISIBLE = {2025: "Feb 03, 2025 5:02pm UTC", 2026: "Sep 16, 2025 9:07am UTC"}
# Values transcribed from the two reviewed official tables, including UTC day rollover.
_CASES = [
    (
        2025,
        1,
        "albert_park",
        "Australian Grand Prix",
        ["Australia", "March 16", "15:00", "04:00"],
        "2025-03-16T04:00:00Z",
    ),
    (
        2025,
        2,
        "shanghai",
        "Chinese Grand Prix",
        ["China", "March 23", "15:00", "07:00"],
        "2025-03-23T07:00:00Z",
    ),
    (
        2025,
        21,
        "interlagos",
        "São Paulo Grand Prix",
        ["Sao Paulo", "November 9", "14:00", "17:00"],
        "2025-11-09T17:00:00Z",
    ),
    (
        2025,
        22,
        "vegas",
        "Las Vegas Grand Prix",
        ["Las Vegas", "November 22", "20:00", "04:00"],
        "2025-11-23T04:00:00Z",
    ),
    (
        2025,
        23,
        "losail",
        "Qatar Grand Prix",
        ["Qatar", "November 30", "19:00", "16:00"],
        "2025-11-30T16:00:00Z",
    ),
    (
        2025,
        24,
        "yas_marina",
        "Abu Dhabi Grand Prix",
        ["Abu Dhabi", "December 7", "17:00", "13:00"],
        "2025-12-07T13:00:00Z",
    ),
    (
        2026,
        1,
        "albert_park",
        "Australian Grand Prix",
        ["Australia, Mar 8", "-", "1600", "1500"],
        "2026-03-08T04:00:00Z",
    ),
    (
        2026,
        2,
        "shanghai",
        "Chinese Grand Prix",
        ["China, Mar 15", "1100", "1500", "1500"],
        "2026-03-15T07:00:00Z",
    ),
    (
        2026,
        3,
        "suzuka",
        "Japanese Grand Prix",
        ["Japan, Mar 29", "-", "1500", "1400"],
        "2026-03-29T05:00:00Z",
    ),
]


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _html(season: int, rows: list[list[str]], modified: str | None = None) -> str:
    article = {
        "@type": "NewsArticle",
        "url": ARTICLE_URLS[season],
        "datePublished": _STAMP[season],
        "dateModified": modified or _STAMP[season],
        "headline": _HEADLINES[season],
    }
    table = "".join(
        "<tr>" + "".join(f"<td><span>{cell}</span></td>" for cell in row) + "</tr>"
        for row in [_HEADERS[season], *rows]
    )
    return (
        f"<h1>{_HEADLINES[season]}</h1><time>{_VISIBLE[season]}</time>"
        f'<script type="application/ld+json">{json.dumps(article)}</script>'
        f"<table>{table}</table>"
        '<script>{"unrelatedStoryUpdatedAt":"2026-09-28T12:00:00Z"}</script>'
    )


def _claims(case: tuple) -> dict:
    season, round_number, circuit, event, _, race_start = case
    return {
        "season": season,
        "round_number": round_number,
        "circuit_id": circuit,
        "event_name": event,
        "claimed_publication": _dt(_STAMP[season]).replace(second=0, microsecond=0),
        "claimed_race_start": _dt(race_start),
        "source_url": SCHEDULE_URLS[season],
        "prediction_timestamp": _dt(race_start).replace(day=1),
    }


@pytest.mark.parametrize("case", _CASES)
def test_reviewed_rows_match_utc_schedule(case: tuple) -> None:
    schedule = validate_f1_schedule(_html(case[0], [case[4]]), **_claims(case))
    assert schedule.race_start == _dt(case[5])
    assert schedule.source_row == tuple(case[4])
    assert schedule.publication_at == _dt(_STAMP[case[0]])
    assert schedule.available_by == schedule.publication_at.replace(
        second=0, microsecond=0
    ).replace(minute=schedule.publication_at.minute + 1)
    assert schedule.publication_precision_seconds == 60


@pytest.mark.parametrize("change", ["clock", "date", "visible", "publication", "heading", "schema"])
def test_changed_source_claims_are_rejected(change: str) -> None:
    case = _CASES[5]
    html = _html(2025, [case[4]])
    replacements = {
        "clock": ("13:00", "14:00"),
        "date": ("December 7", "December 8"),
        "visible": ("5:02pm UTC", "5:03pm UTC"),
        "publication": ("2025-02-03T17:02:27.315Z", "2025-02-04T17:02:27.315Z"),
        "heading": (_HEADLINES[2025], "Race start times for 2024"),
        "schema": ("LOCAL START TIME", "CURRENT START TIME"),
    }
    before, after = replacements[change]
    with pytest.raises(ScheduleValidationError):
        validate_f1_schedule(html.replace(before, after), **_claims(case))


def test_duplicate_event_rows_are_rejected_even_when_identical() -> None:
    case = _CASES[5]
    with pytest.raises(ScheduleValidationError, match="duplicated"):
        validate_f1_schedule(_html(2025, [case[4], case[4]]), **_claims(case))


@pytest.mark.parametrize(
    "field,value",
    [
        ("circuit_id", "bahrain"),
        ("event_name", "Bahrain Grand Prix"),
        ("round_number", 23),
        ("source_url", "https://example.com/schedule"),
        ("claimed_race_start", _dt("2025-12-07T14:00:00Z")),
        ("claimed_publication", datetime(2025, 2, 3, 17, 2)),
    ],
)
def test_unreviewed_or_conflicting_claims_are_rejected(field: str, value: object) -> None:
    case = _CASES[5]
    claims = _claims(case)
    claims[field] = value
    with pytest.raises(ScheduleValidationError):
        validate_f1_schedule(_html(2025, [case[4]]), **claims)


def test_article_modified_after_cutoff_is_rejected() -> None:
    case = _CASES[5]
    with pytest.raises(ScheduleValidationError, match="not known"):
        validate_f1_schedule(
            _html(2025, [case[4]], modified="2025-12-08T10:00:00Z"), **_claims(case)
        )


def test_modified_article_before_cutoff_uses_modification_availability() -> None:
    case = _CASES[5]
    result = validate_f1_schedule(
        _html(2025, [case[4]], modified="2025-12-01T00:00:00Z"),
        **{**_claims(case), "prediction_timestamp": _dt("2025-12-06T15:17:00Z")},
    )
    assert result.available_by == _dt("2025-12-01T00:01:00Z")


def test_minute_resolution_requires_end_of_publication_minute() -> None:
    case = _CASES[5]
    claims = {**_claims(case), "prediction_timestamp": _dt("2025-02-03T17:02:59Z")}
    with pytest.raises(ScheduleValidationError, match="not known"):
        validate_f1_schedule(_html(2025, [case[4]]), **claims)


def test_canonical_single_id_alias_is_supported_for_2026() -> None:
    case = _CASES[8]
    schedule = validate_f1_schedule(
        _html(2026, [case[4]]), **{**_claims(case), "source_url": ARTICLE_URLS[2026]}
    )
    assert schedule.timezone_basis == "Asia/Tokyo"
    assert schedule.race_start.tzinfo == UTC


def test_legacy_2022_article_uses_later_jsonld_time_and_exact_utc_row() -> None:
    publication = "2022-02-11T17:13:44.277Z"
    article = {
        "@type": "NewsArticle",
        "url": SCHEDULE_URLS[2022],
        "datePublished": publication,
        "dateModified": "2022-02-11T00:00:00Z",
        "headline": "2022 F1 Grand Prix start times confirmed",
    }
    html = (
        "<h1>2022 F1 Grand Prix start times confirmed</h1>"
        "<time>Feb 11, 2022 12:00am UTC</time>"
        f'<script type="application/ld+json">{json.dumps(article)}</script>'
        "<table><tr><th>GRAND PRIX</th><th>DATE</th><th>LOCAL TIME</th><th>UTC</th></tr>"
        "<tr><td>Bahrain</td><td>March 20</td><td>1800</td><td>1500</td></tr></table>"
    )
    claims = {
        "season": 2022,
        "round_number": 1,
        "event_name": "Bahrain Grand Prix",
        "circuit_id": "bahrain",
        "claimed_publication": _dt(publication),
        "claimed_race_start": _dt("2022-03-20T15:00:00Z"),
        "source_url": SCHEDULE_URLS[2022],
        "prediction_timestamp": _dt("2022-03-19T15:00:00Z"),
    }
    result = validate_f1_schedule(html, **claims)
    assert result.available_by == _dt("2022-02-11T17:14:00Z")
    with pytest.raises(ScheduleValidationError, match="GMT clock"):
        validate_f1_schedule(html.replace("1500</td>", "1600</td>"), **claims)


def test_historical_event_timetable_binds_published_race_row() -> None:
    url = (
        "https://www.formula1.com/en/latest/article/"
        "formula-1-gulf-air-bahrain-grand-prix-2023-timetable.fixture123"
    )
    heading = "FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2023"
    article = {
        "@type": "NewsArticle",
        "@id": url,
        "url": url,
        "headline": heading + " - full timetable | Formula 1",
        "datePublished": "2023-01-27T11:33:43.263Z",
        "dateModified": "2022-01-01T00:00:00Z",
    }
    html = (
        f"<h1>{heading}</h1>"
        f'<script type="application/ld+json">{json.dumps(article)}</script>'
        "<table><tr><th>SUNDAY 5th MARCH</th><th></th><th></th></tr>"
        "<tr><td>FORMULA 1</td><td>GRAND PRIX (57 LAPS OR 120 MINS)</td>"
        "<td>18:00 - 20:00</td></tr></table>"
    )
    claims = {
        "season": 2023,
        "round_number": 1,
        "event_name": "Bahrain Grand Prix",
        "circuit_id": "bahrain",
        "claimed_publication": _dt("2023-01-27T11:33:43.263Z"),
        "claimed_race_start": _dt("2023-03-05T15:00:00Z"),
        "source_url": url,
        "prediction_timestamp": _dt("2023-03-04T15:00:00Z"),
    }
    result = validate_event_timetable(html, **claims)
    assert result.available_by == _dt("2023-01-27T11:34:00Z")
    with pytest.raises(ScheduleValidationError, match="contradicts claimed UTC"):
        validate_event_timetable(html.replace("18:00 - 20:00", "19:00 - 21:00"), **claims)
    with pytest.raises(ScheduleValidationError, match="event identity"):
        validate_event_timetable(
            html.replace("BAHRAIN GRAND PRIX", "AUSTRALIAN GRAND PRIX"), **claims
        )
    with pytest.raises(ScheduleValidationError, match="not known by cutoff"):
        validate_event_timetable(
            html.replace("2022-01-01T00:00:00Z", "2023-03-06T00:00:00Z"), **claims
        )


def test_overnight_local_timetable_uses_next_utc_day() -> None:
    url = (
        "https://www.formula1.com/en/latest/article/"
        "formula-1-las-vegas-grand-prix-2024-timetable.fixture123"
    )
    heading = "FORMULA 1 LAS VEGAS GRAND PRIX 2024"
    article = {
        "@type": "NewsArticle",
        "@id": url,
        "url": url,
        "headline": heading + " - full timetable | Formula 1",
        "datePublished": "2024-10-14T10:00:00Z",
    }
    html = (
        f"<h1>{heading}</h1>"
        f'<script type="application/ld+json">{json.dumps(article)}</script>'
        "<table><tr><th>SATURDAY 23rd NOVEMBER</th><th></th><th></th></tr>"
        "<tr><td>FORMULA 1</td><td>GRAND PRIX (50 LAPS OR 120 MINS)</td>"
        "<td>22:00 - 00:00</td></tr></table>"
    )
    claims = {
        "season": 2024,
        "round_number": 22,
        "event_name": "Las Vegas Grand Prix",
        "circuit_id": "vegas",
        "claimed_publication": _dt("2024-10-14T10:00:00Z"),
        "claimed_race_start": None,
        "source_url": url,
        "prediction_timestamp": _dt("2024-11-23T08:00:00Z"),
    }
    result = validate_event_timetable(html, **claims)
    assert result.race_start == _dt("2024-11-24T06:00:00Z")


def test_fia_amendment_supersedes_original_schedule_before_cutoff() -> None:
    text = (
        "2024 SAO PAULO GRAND PRIX\n"
        "The stewards approve Version 5 of the timetable.\n"
        "SUNDAY 03 NOVEMBER 2024\n"
        "12:30 14:30 FORMULA 1 TRACK GRAND PRIX\n"
    )
    claims = {
        "season": 2024,
        "round_number": 21,
        "event_name": "São Paulo Grand Prix",
        "circuit_id": "interlagos",
        "claimed_publication": _dt("2024-11-02T21:31:00Z"),
        "claimed_race_start": _dt("2024-11-03T15:30:00Z"),
        "source_url": "https://www.fia.com/decision/amended-timetable.pdf",
        "prediction_timestamp": _dt("2024-11-03T10:30:00Z"),
        "document_sha256": "a" * 64,
    }
    assert (
        validate_fia_timetable_amendment(text, **claims).race_start == claims["claimed_race_start"]
    )
    with pytest.raises(ScheduleValidationError, match="not available at cutoff"):
        validate_fia_timetable_amendment(
            text, **{**claims, "prediction_timestamp": _dt("2024-11-02T21:31:00Z")}
        )
    with pytest.raises(ScheduleValidationError, match="approval"):
        validate_fia_timetable_amendment(text.replace("approve", "reject"), **claims)


_SEPANG = RACE_TIME_ARTICLES[(2026, 16)]


def _race_time_html(
    *,
    modified: str = "2026-09-29T20:56:06.110Z",
    sentence: str = "the Grand Prix itself gets underway at 1500 on Sunday, October 4.",
) -> str:
    article = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "url": _SEPANG["url"],
        "headline": _SEPANG["headline"],
        "datePublished": "2026-09-29T20:56:06.110Z",
        "dateModified": modified,
    }
    return (
        f'<script type="application/ld+json">{json.dumps(article)}</script>'
        f"<h1>{_SEPANG['headline']}</h1><time>Sep 29, 2026 8:56pm UTC</time>"
        f"<p>Qualifying at 1600 before {sentence}</p>"
    )


def _race_time(html: str, **changes: object):
    claim = {
        "season": 2026,
        "round_number": 16,
        "event_name": "Bahrain Grand Prix in Malaysia",
        "circuit_id": "sepang",
        "claimed_publication": datetime(2026, 9, 29, 20, 56, 6, 110000, tzinfo=UTC),
        "claimed_race_start": datetime(2026, 10, 4, 7, tzinfo=UTC),
        "source_url": _SEPANG["url"],
        "prediction_timestamp": datetime(2026, 10, 3, 9, tzinfo=UTC),
    }
    claim.update(changes)
    return validate_race_time_article(html, **claim)


def test_relocated_race_time_article_binds_local_start() -> None:
    verified = _race_time(_race_time_html())
    assert verified.race_start == datetime(2026, 10, 4, 7, tzinfo=UTC)
    assert verified.available_by == datetime(2026, 9, 29, 20, 57, tzinfo=UTC)
    assert verified.timezone_basis == "Asia/Kuala_Lumpur"


@pytest.mark.parametrize(
    ("html", "changes"),
    [
        (_race_time_html(modified="2022-01-01T00:00:00.000Z"), {}),
        (_race_time_html(sentence="the race starts at 1500 on Sunday, October 4."), {}),
        (
            _race_time_html(
                sentence="the Grand Prix itself gets underway at 1500 on Monday, October 4."
            ),
            {},
        ),
        (_race_time_html(), {"claimed_race_start": datetime(2026, 10, 4, 12, tzinfo=UTC)}),
        (_race_time_html(), {"prediction_timestamp": datetime(2026, 9, 29, 20, 56, tzinfo=UTC)}),
        (_race_time_html(), {"round_number": 17}),
        (_race_time_html(), {"source_url": _SEPANG["url"] + "x"}),
    ],
)
def test_race_time_article_rejects_unreviewed_or_inconsistent_claims(
    html: str, changes: dict
) -> None:
    with pytest.raises(ScheduleValidationError):
        _race_time(html, **changes)
