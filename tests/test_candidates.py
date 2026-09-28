import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from f1_ml_predictor.trust.candidates import FIA_ROOT, discover_candidates


def _race(round_number: int, *, date: str = "2025-03-02", name: str | None = None) -> dict:
    return {
        "season": "2025",
        "round": str(round_number),
        "raceName": name or f"Race {round_number} Grand Prix",
        "Circuit": {"circuitId": f"circuit_{round_number}"},
        "date": date,
        "time": "12:00:00Z",
        "Qualifying": {"date": date},
    }


def _selectors(options: list[tuple[str, str]]) -> str:
    return (
        "<select>"
        + "".join(f'<option value="{url}">{label}</option>' for url, label in options)
        + "</select>"
    )


def _document(
    number: int, title: str, slug: str, *, date: str = "01.03.25 10:00", recalled: bool = False
) -> str:
    link = f'<a href="/system/files/decision-document/{slug}">Doc {number} - {title}</a>'
    if recalled:
        link = f"Doc {number} - {title} Recalled"
    return f'<li class="document-row">{link} Published on {date} CET</li>'


def _client(
    races: list[dict],
    *,
    duplicate_selector: bool = False,
    wrong_year: bool = False,
    missing_provisional: bool = False,
    bad_pagination: bool = False,
) -> tuple[httpx.Client, list[str]]:
    seen: list[str] = []
    season_path = (
        "/documents/championships/fia-formula-one-world-championship-14/season/season-2025-9999"
    )
    options = [
        (season_path + "/event/" + row["raceName"].replace(" ", "%20"), row["raceName"])
        for row in races
    ]
    if duplicate_selector and options:
        options.append(options[0])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "api.jolpi.ca":
            assert dict(request.url.params) == {"limit": "100", "offset": "0"}
            return httpx.Response(
                200,
                json={
                    "MRData": {
                        "total": str(len(races) + int(bad_pagination)),
                        "limit": "100",
                        "offset": "0",
                        "RaceTable": {"season": "2025", "Races": races},
                    }
                },
                headers={"etag": "schedule-v1"},
            )
        if str(request.url) == FIA_ROOT:
            return httpx.Response(200, text=_selectors([(season_path, "SEASON 2025")]))
        if request.url.path == season_path:
            return httpx.Response(200, text=_selectors(options))
        for row in races:
            if request.url.path == season_path + "/event/" + row["raceName"]:
                year = 2024 if wrong_year else 2025
                slug = f"{year}_" + row["raceName"].lower().replace(" ", "_") + "_-_"
                clock = "01.07.25 10:00" if "-07-" in row["date"] else "01.03.25 10:00"
                provisional = "Final" if missing_provisional else "Provisional"
                html = _document(
                    23,
                    f"{provisional} Qualifying Classification",
                    slug + "provisional_qualifying_classification.pdf",
                    date=clock,
                )
                html += _document(11, "Entry List", slug + "entry_list.pdf", date=clock)
                html += _document(
                    48,
                    "Final Race Classification",
                    slug + "final_race_classification.pdf",
                    date=clock,
                )
                html += _document(
                    46,
                    "Provisional Qualifying Classification",
                    "unused.pdf",
                    date=clock,
                    recalled=True,
                )
                return httpx.Response(200, text=html)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_automatic_pool_retains_only_bounded_metadata_and_never_gold(tmp_path: Path) -> None:
    client, seen = _client([_race(number) for number in range(1, 25)])
    with client:
        result = discover_candidates(
            tmp_path, seasons=(2025,), http_client=client, now=datetime(2026, 1, 1, tzinfo=UTC)
        )
    catalog = result["catalog"]
    assert len(catalog["candidates"]) == 17
    assert len(catalog["excluded_events"]) == 7
    assert len(seen) == 20  # schedule, root, season, seventeen event registries
    assert not any(".pdf" in url for url in seen)
    assert all(not row["eligible_gold"] for row in catalog["candidates"])
    assert result["auditability_report"]["eligible_gold_races"] == 0
    assert result["auditability_report"]["preferred_pool_size_met"]
    artifact = next(item for item in catalog["source_artifacts"] if item["provider"] == "Jolpica")
    assert artifact["request_parameters"] == {"limit": "100", "offset": "0"}
    assert artifact["provider_version"] == "schedule-v1"
    assert artifact["evidence_classification"] == "current_state_only"
    for item in catalog["source_artifacts"]:
        assert hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    saved = tmp_path / result["catalog_path"]
    assert json.loads(saved.read_text()) == catalog
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == result["catalog_sha256"]
    assert (
        hashlib.sha256((tmp_path / result["report_path"]).read_bytes()).hexdigest()
        == result["report_sha256"]
    )


def test_winter_evidence_ranks_before_more_recent_summer(tmp_path: Path) -> None:
    client, _ = _client([_race(1), _race(12, date="2025-07-02")])
    with client:
        result = discover_candidates(tmp_path, (2025,), 2, client, datetime(2026, 1, 1, tzinfo=UTC))
    first, second = result["catalog"]["candidates"]
    assert first["round"] == 1 and first["winter_publication_record"]
    assert second["published_at_utc"] is None
    assert second["prediction_timestamp_utc"] is None
    assert "summer_publication_clock_unresolved" in second["exclusion_reasons"]
    assert first["qualifying_url"].endswith("provisional_qualifying_classification.pdf")
    assert first["unbound_recalled_records"][0]["recalled"]
    assert "recalled_record_event_binding_requires_review" in first["exclusion_reasons"]


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"duplicate_selector": True}, "selector is missing or ambiguous"),
        ({"wrong_year": True}, "does_not_match_exact_event_and_season"),
        ({"missing_provisional": True}, "provisional_qualifying_record_missing"),
    ],
)
def test_uncertain_selector_or_registry_is_excluded(
    tmp_path: Path, kwargs: dict, reason: str
) -> None:
    client, _ = _client([_race(1)], **kwargs)
    with client:
        result = discover_candidates(tmp_path, (2025,), 1, client, datetime(2026, 1, 1, tzinfo=UTC))
    candidate = result["catalog"]["candidates"][0]
    assert candidate["status"] == "excluded"
    assert reason in candidate["exclusion_reasons"][0]
    assert not candidate["eligible_gold"]


def test_future_races_and_duplicate_rounds_never_enter_pool(tmp_path: Path) -> None:
    client, seen = _client([_race(1), _race(1), _race(2, date="2026-12-01")])
    with client:
        result = discover_candidates(
            tmp_path, (2025,), 17, client, datetime(2026, 1, 1, tzinfo=UTC)
        )
    assert not result["catalog"]["candidates"]
    assert {row["reason"] for row in result["catalog"]["excluded_events"]} == {
        "duplicate_schedule_round",
        "race_not_completed_as_of_discovery",
    }
    assert len(seen) == 2  # schedule plus championship root, no event registries
    assert result["auditability_report"]["status"] == "insufficient_data"


def test_incomplete_schedule_is_explicit_source_error(tmp_path: Path) -> None:
    client, _ = _client([_race(1)], bad_pagination=True)
    with client:
        result = discover_candidates(
            tmp_path, (2025,), 17, client, datetime(2026, 1, 1, tzinfo=UTC)
        )
    assert not result["catalog"]["candidates"]
    assert "pagination is incomplete" in result["catalog"]["source_errors"][0]["reason"]
    assert any(item["provider"] == "Jolpica" for item in result["catalog"]["source_artifacts"])


@pytest.mark.parametrize("limit", [0, 21, True])
def test_discovery_limit_is_bounded(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        discover_candidates(tmp_path, limit=limit)


def test_naive_discovery_time_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        discover_candidates(tmp_path, now=datetime(2026, 1, 1))


def test_qualifying_revision_at_proposed_cutoff_requires_review(tmp_path: Path) -> None:
    client, _ = _client([_race(1)])
    original_transport = client._transport

    def handler(request: httpx.Request) -> httpx.Response:
        response = original_transport.handle_request(request)
        if "/event/" in request.url.path:
            slug = "2025_race_1_grand_prix_-_final_qualifying_classification.pdf"
            response = httpx.Response(
                200,
                text=response.text
                + _document(24, "Final Qualifying Classification", slug, date="01.03.25 10:02"),
            )
        return response

    with httpx.Client(transport=httpx.MockTransport(handler)) as modified:
        result = discover_candidates(
            tmp_path, (2025,), 1, modified, datetime(2026, 1, 1, tzinfo=UTC)
        )
    candidate = result["catalog"]["candidates"][0]
    assert len(candidate["qualifying_versions"]) == 2
    assert "another_qualifying_version_at_proposed_cutoff" in candidate["exclusion_reasons"]
    assert not candidate["eligible_gold"]


def test_failed_season_selector_is_requested_once_for_seventeen_candidates(tmp_path: Path) -> None:
    template, _ = _client([_race(number) for number in range(1, 18)])
    season_url = FIA_ROOT + "/season/season-2025-9999"
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if str(request.url) == season_url:
            raise httpx.ReadTimeout("FIA season selector timed out", request=request)
        return template._transport.handle_request(request)

    with template, httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = discover_candidates(
            tmp_path, (2025,), 17, client, datetime(2026, 1, 1, tzinfo=UTC)
        )

    assert requested.count(season_url) == 1
    assert len(requested) == 3  # schedule, championship root, failed season page
    candidates = result["catalog"]["candidates"]
    assert len(candidates) == 17
    assert all(candidate["status"] == "excluded" for candidate in candidates)
    assert all(
        candidate["exclusion_reasons"] == ["FIA season selector timed out"]
        for candidate in candidates
    )
    assert not any(candidate["eligible_gold"] for candidate in candidates)
    assert result["auditability_report"]["candidates_with_publication_records"] == 0
    assert result["auditability_report"]["status"] == "insufficient_data"
