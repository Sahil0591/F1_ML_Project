import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
import pytest
from test_collected_features import EVENT, raw_bundle

from f1_ml_predictor.trust import collected_outcomes, scheduler
from f1_ml_predictor.trust.collected_features import certify_capture
from f1_ml_predictor.trust.collected_outcomes import collect_final_outcomes
from f1_ml_predictor.trust.prospective import verify_bundle
from f1_ml_predictor.trust.scheduler import import_scheduler_outcomes, scheduler_tick

OBSERVED = datetime(2026, 10, 5, 9, tzinfo=UTC)
ROOT_URL = "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14"
SEASON_URL = ROOT_URL + "/season/season-2026-2072"
REGISTRY_URL = SEASON_URL + "/event/Bahrain%20Grand%20Prix"
PDF_URL = (
    "https://www.fia.com/sites/default/files/decision-document/"
    "2026_bahrain_grand_prix_-_final_race_classification.pdf"
)
TEXT = """2026 BAHRAIN GRAND PRIX
Document 71
Final Race Classification
POS NO DRIVER LAPS TIME GAP
1 11 Driver A Team A 57 1:30:00.000
2 22 Driver B Team A 57 +1.000
"""


def options(*groups: list[tuple[str, str]]) -> str:
    return "".join(
        "<select>"
        + "".join(f'<option value="{url}">{title}</option>' for url, title in group)
        + "</select>"
        for group in groups
    )


def document(
    *,
    identifier: str = "71",
    title: str = "Final Race Classification",
    published: str = "04.10.26 18:11",
    url: str = PDF_URL,
    recalled: bool = False,
) -> str:
    return (
        '<li class="document-row">'
        f'<a href="{url}">Doc {identifier} - {title}</a>'
        f" Published on {published} CET" + (" RECALLED" if recalled else "") + "</li>"
    )


def registry(*rows: str, season: int = 2026, race: str = "Bahrain Grand Prix") -> str:
    return options([(ROOT_URL, f"SEASON {season}")], [(SEASON_URL, race)]) + "".join(
        rows or (document(),)
    )


def fia_client(
    *,
    seen: list[httpx.Request] | None = None,
    first_registry: str | None = None,
    second_registry: str | None = None,
    root_page: str | None = None,
    season_page: str | None = None,
    pdf: bytes = b"%PDF-1.7 retained mocked FIA bytes",
    other: Callable[[httpx.Request], httpx.Response] | None = None,
) -> httpx.Client:
    count = 0

    def response(request: httpx.Request) -> httpx.Response:
        nonlocal count
        if seen is not None:
            seen.append(request)
        url = str(request.url)
        if request.url.host != "www.fia.com":
            assert other is not None
            return other(request)
        if url == ROOT_URL:
            text = root_page or options([("0", "Season"), (SEASON_URL, "SEASON 2026")])
        elif url == SEASON_URL:
            text = season_page or options(
                [(ROOT_URL, "SEASON 2026")], [(REGISTRY_URL, "Bahrain Grand Prix")]
            )
        elif url == REGISTRY_URL:
            count += 1
            text = (
                first_registry or registry()
                if count == 1
                else second_registry or first_registry or registry()
            )
        elif url == PDF_URL:
            return httpx.Response(
                200,
                content=pdf,
                headers={
                    "content-type": "application/pdf",
                    "etag": '"final-v71"',
                    "last-modified": "Sun, 04 Oct 2026 17:11:00 GMT",
                },
            )
        else:
            raise AssertionError(f"unexpected official request: {url}")
        return httpx.Response(200, text=text, headers={"content-type": "text/html"})

    return httpx.Client(transport=httpx.MockTransport(response))


@pytest.fixture
def inspected(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    result: dict[str, Any] = {
        "document_id": "71",
        "pages": 1,
        "text": TEXT,
        "extraction_version": "mock-layout-v1",
    }
    monkeypatch.setattr(collected_outcomes, "inspect_pdf", lambda path: dict(result))
    monkeypatch.setattr(
        collected_outcomes, "DRIVER_ALIASES", {"Driver A": "driver_a", "Driver B": "driver_b"}
    )
    monkeypatch.setattr(collected_outcomes, "CONSTRUCTOR_ALIASES", {"Team A": "team_a"})
    return result


def collect(root: Path, bundle: Path, client: httpx.Client) -> dict[str, Any]:
    return collect_final_outcomes(root, bundle, http_client=client, now=lambda: OBSERVED)


def seed_status(root: Path) -> dict[str, Any]:
    bundle = raw_bundle(root)
    certified = certify_capture(root, bundle)
    manifest = verify_bundle(bundle)
    state = {
        "version": 1,
        "status": "captured",
        "evaluation_eligible": False,
        "events": {
            EVENT.partition(): {
                "season": EVENT.season,
                "round": EVENT.round,
                "race_name": "Bahrain Grand Prix",
                "race_start": "2026-10-04T09:00:00+00:00",
                "captures": [
                    {
                        "bundle": bundle.relative_to(root).as_posix(),
                        "manifest_sha256": manifest["manifest_sha256"],
                        "cutoff_kind": "post_qualifying",
                        "capture_version": 1,
                        "captured_at": manifest["captured_at"],
                        **certified,
                    }
                ],
            }
        },
    }
    path = root / "data/raw/prospective_scheduler/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    return state


def test_captures_latest_final_with_retained_hashes_and_unknown_dnf(
    tmp_path: Path, inspected: dict[str, Any]
) -> None:
    bundle = raw_bundle(tmp_path)
    original = {file.name: file.read_bytes() for file in bundle.iterdir()}
    seen: list[httpx.Request] = []
    records = registry(
        document(identifier="69", published="04.10.26 17:11", recalled=True),
        document(),
        document(identifier="72", title="Championship Points", published="04.10.26 18:12"),
    )
    with fia_client(seen=seen, first_registry=records) as client:
        result = collect(tmp_path, bundle, client)
    assert [str(request.url) for request in seen] == [
        ROOT_URL,
        SEASON_URL,
        REGISTRY_URL,
        PDF_URL,
        REGISTRY_URL,
    ]
    table = pq.ParquetFile(tmp_path / result["path"]).read()
    assert table["driver_id"].to_pylist() == ["driver_a", "driver_b"]
    assert table["winner"].to_pylist() == [True, False]
    assert table["dnf"].to_pylist() == [None, None]
    assert table["dnf_category"].to_pylist() == ["unknown", "unknown"]
    assert table["label_available_at"].to_pylist() == [OBSERVED, OBSERVED]
    proof_bytes = (tmp_path / result["audit"]["path"]).read_bytes()
    assert hashlib.sha256(proof_bytes).hexdigest() == result["audit"]["sha256"]
    proof = json.loads(proof_bytes)
    assert proof["capture_manifest_sha256"] == verify_bundle(bundle)["manifest_sha256"]
    assert proof["document_id"] == "71"
    assert proof["conservative_publication_bound_utc"] == "2026-10-04T17:12:00+00:00"
    assert proof["source_observations"][3]["provider_version"] == '"final-v71"'
    for source in proof["source_observations"]:
        assert source["provider"] == "FIA"
        assert source["observed_at_utc"] == OBSERVED.isoformat()
        assert (
            source["sha256"] == hashlib.sha256((tmp_path / source["path"]).read_bytes()).hexdigest()
        )
    assert original == {file.name: file.read_bytes() for file in bundle.iterdir()}
    assert not (tmp_path / "data/benchmarks/prospective_registry.json").exists()


@pytest.mark.parametrize(
    "bad_registry",
    [
        registry(document(recalled=True)),
        registry(document(title="Provisional Race Classification")),
        registry(document(title="Final Sprint Race Classification")),
        registry(document(url=PDF_URL.replace("2026_", "2025_"))),
        registry(
            document(), document(identifier="72", title="Infringement", published="04.10.26 18:12")
        ),
        registry(document(), document(identifier="72", title="Infringement")),
        registry(
            document(),
            document(identifier="72", title="Infringement", url="https://www.fia.com/unknown.pdf"),
        ),
        registry(
            document(),
            '<li class="document-row">Doc 72 - Infringement without a resolved clock</li>',
        ),
        registry(document(), document()),
        registry(season=2025),
        registry(race="Singapore Grand Prix"),
        registry(document(published="03.10.26 18:11")),
        registry(document(published="06.10.26 18:11")),
    ],
)
def test_uncertain_event_or_final_versions_never_emit_labels(
    tmp_path: Path, inspected: dict[str, Any], bad_registry: str
) -> None:
    bundle = raw_bundle(tmp_path)
    seen: list[httpx.Request] = []
    with fia_client(seen=seen, first_registry=bad_registry) as client:
        with pytest.raises(ValueError):
            collect(tmp_path, bundle, client)
    assert len(seen) <= 5
    assert not list(
        (tmp_path / "data/raw/prospective_scheduler/automatic_outcomes").rglob("*.parquet")
    )


@pytest.mark.parametrize(
    "change",
    ["document_id", "header_year", "header_event", "driver", "constructor", "incomplete_roster"],
)
def test_pdf_identity_and_full_entrant_roster_must_match_frozen_capture(
    tmp_path: Path, inspected: dict[str, Any], change: str
) -> None:
    if change == "document_id":
        inspected["document_id"] = "70"
    elif change == "header_year":
        inspected["text"] = TEXT.replace("2026", "2025")
    elif change == "header_event":
        inspected["text"] = TEXT.replace("BAHRAIN", "SINGAPORE")
    elif change == "driver":
        inspected["text"] = TEXT.replace("Driver B", "Unmapped Driver")
    elif change == "constructor":
        inspected["text"] = TEXT.replace("Team A", "Team B")
    else:
        inspected["text"] = TEXT.replace("2 22 Driver B Team A 57 +1.000\n", "")
    with fia_client() as client, pytest.raises(ValueError):
        collect(tmp_path, raw_bundle(tmp_path), client)
    assert not list(
        (tmp_path / "data/raw/prospective_scheduler/automatic_outcomes").rglob("*.parquet")
    )


def test_registry_recheck_rejects_a_new_ruling_while_capturing_pdf(
    tmp_path: Path, inspected: dict[str, Any]
) -> None:
    changed = registry(
        document(), document(identifier="72", title="Decision", published="04.10.26 18:12")
    )
    seen: list[httpx.Request] = []
    with fia_client(seen=seen, second_registry=changed) as client:
        with pytest.raises(ValueError, match="later"):
            collect(tmp_path, raw_bundle(tmp_path), client)
    assert len(seen) == 5


def test_no_network_before_retained_race_start(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    with fia_client(seen=seen) as client, pytest.raises(ValueError, match="after race start"):
        collect_final_outcomes(
            tmp_path,
            raw_bundle(tmp_path),
            http_client=client,
            now=lambda: datetime(2026, 10, 4, 9, tzinfo=UTC),
        )
    assert not seen


def test_discovered_event_routes_must_be_official_exact_season_event(
    tmp_path: Path, inspected: dict[str, Any]
) -> None:
    for url in (
        "https://other.example/event/Bahrain%20Grand%20Prix",
        SEASON_URL.replace("2026", "2025") + "/event/Bahrain%20Grand%20Prix",
        SEASON_URL + "/event/Singapore%20Grand%20Prix",
    ):
        page = options([(ROOT_URL, "SEASON 2026")], [(url, "Bahrain Grand Prix")])
        with fia_client(season_page=page) as client, pytest.raises(ValueError):
            collect(tmp_path, raw_bundle(tmp_path), client)


def test_automatic_import_verifies_audit_and_source_hashes(
    tmp_path: Path, inspected: dict[str, Any]
) -> None:
    state = seed_status(tmp_path)
    capture = state["events"][EVENT.partition()]["captures"][0]
    original = (tmp_path / capture["features"]["path"]).read_bytes()
    with fia_client() as client:
        result = collect(tmp_path, tmp_path / capture["bundle"], client)
    proof = json.loads((tmp_path / result["audit"]["path"]).read_text())
    source = tmp_path / proof["source_observations"][3]["path"]
    raw = source.read_bytes()
    source.write_bytes(raw + b"tampered")
    with pytest.raises(ValueError, match="source hash"):
        import_scheduler_outcomes(
            tmp_path,
            tmp_path / result["path"],
            expected_sha256=result["sha256"],
            source_audit=result["audit"],
            now=lambda: OBSERVED,
        )
    source.write_bytes(raw)
    imported = import_scheduler_outcomes(
        tmp_path,
        tmp_path / result["path"],
        expected_sha256=result["sha256"],
        source_audit=result["audit"],
        now=lambda: OBSERVED,
    )
    assert imported["evaluation_eligible"] is True
    assert imported["source_audit"] == result["audit"]
    registry_data = json.loads((tmp_path / "data/benchmarks/prospective_registry.json").read_text())
    outcomes = registry_data["races"][0]["outcomes"]
    assert outcomes["audit_path"] == result["audit"]["path"]
    assert outcomes["audit_sha256"] == result["audit"]["sha256"]
    assert (tmp_path / capture["features"]["path"]).read_bytes() == original
    assert (
        pq.ParquetFile(tmp_path / "data/benchmarks/prospective/gold.parquet").read().num_rows == 2
    )


def test_tick_prioritizes_next_race_then_collects_only_one_oldest_missing_label(
    tmp_path: Path, inspected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = seed_status(tmp_path)
    older = initial["events"][EVENT.partition()]
    initial["events"]["season=2026/round=17"] = {
        "season": 2026,
        "round": 17,
        "race_start": (OBSERVED - timedelta(hours=1)).isoformat(),
        "captures": [{"bundle": "unused-second-event"}],
    }
    scheduler._save(tmp_path, initial)
    order: list[str] = []

    def next_race(root: Path, state: dict[str, Any], **kwargs: Any) -> None:
        order.append("next_race_capture_decision")
        state["status"] = "waiting_for_qualifying"

    original_collect = scheduler.collect_final_outcomes

    def later_labels(root: Path, bundle: Path, **kwargs: Any) -> dict[str, Any]:
        order.append("oldest_labels")
        assert bundle == tmp_path / older["captures"][0]["bundle"]
        return original_collect(root, bundle, **kwargs)

    monkeypatch.setattr(scheduler, "_tick", next_race)
    monkeypatch.setattr(scheduler, "collect_final_outcomes", later_labels)
    seen: list[httpx.Request] = []
    with fia_client(seen=seen) as client:
        state = scheduler_tick(tmp_path, http_client=client, now=lambda: OBSERVED)
        repeated = scheduler_tick(tmp_path, http_client=client, now=lambda: OBSERVED)
    assert state == repeated
    assert order == ["next_race_capture_decision", "oldest_labels"]
    assert len(seen) == 5
    assert state["status"] == "waiting_for_qualifying"
    assert state["events"][EVENT.partition()]["outcome_collection"]["status"] == "captured"
    assert len(state["events"][EVENT.partition()]["outcome_versions"]) == 1
    assert "outcome_collection" not in state["events"]["season=2026/round=17"]
    assert state["evaluation_eligible"] is True


def test_uncertain_labels_defer_without_changing_future_capture_status(
    tmp_path: Path, inspected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_status(tmp_path)

    def next_race(root: Path, state: dict[str, Any], **kwargs: Any) -> None:
        state["status"] = "captured"

    monkeypatch.setattr(scheduler, "_tick", next_race)
    with fia_client(first_registry=registry(document(recalled=True))) as client:
        state = scheduler_tick(tmp_path, http_client=client, now=lambda: OBSERVED)
    entry = state["events"][EVENT.partition()]
    assert state["status"] == "captured"
    assert entry["outcome_collection"]["status"] == "deferred"
    assert entry["outcome_collection"]["reason"]
    assert "outcome_versions" not in entry
    assert state["evaluation_eligible"] is False


def test_full_tick_joins_retained_features_after_next_event_schedule_check(
    tmp_path: Path, inspected: dict[str, Any]
) -> None:
    initial = seed_status(tmp_path)
    original = initial["events"][EVENT.partition()]["captures"][0]
    feature_bytes = (tmp_path / original["features"]["path"]).read_bytes()
    future = {
        "season": "2026",
        "round": "17",
        "raceName": "Next Grand Prix",
        "Circuit": {"circuitId": "next_circuit"},
        "date": "2026-10-18",
        "time": "09:00:00Z",
        "Qualifying": {"date": "2026-10-17", "time": "09:00:00Z"},
    }

    def jolpica(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.jolpi.ca"
        assert request.url.path == "/ergast/f1/2026/"
        return httpx.Response(
            200,
            json={
                "MRData": {
                    "limit": "100",
                    "offset": "0",
                    "total": "1",
                    "RaceTable": {"Races": [future]},
                }
            },
        )

    seen: list[httpx.Request] = []
    with fia_client(seen=seen, other=jolpica) as client:
        first = scheduler_tick(tmp_path, http_client=client, now=lambda: OBSERVED)
        later = scheduler_tick(
            tmp_path, http_client=client, now=lambda: OBSERVED + timedelta(seconds=301)
        )
    assert seen[0].url.host == "api.jolpi.ca"
    assert len([request for request in seen if request.url.host == "www.fia.com"]) == 5
    assert first["active_event"] == "season=2026/round=17"
    assert first["status"] == later["status"] == "waiting_for_qualifying"
    event = later["events"][EVENT.partition()]
    assert event["status"] == "closed"
    assert len(event["outcome_versions"]) == 1
    assert event["captures"][0]["evaluation_eligible"] is True
    assert (tmp_path / original["features"]["path"]).read_bytes() == feature_bytes
    labels = pq.ParquetFile(tmp_path / "data/benchmarks/prospective/gold.parquet").read()
    assert labels["label_winner"].to_pylist() == [True, False]


def test_pdf_parser_errors_are_deferred_without_stopping_the_tick(
    tmp_path: Path, inspected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_status(tmp_path)
    monkeypatch.setattr(scheduler, "_tick", lambda root, state, **kwargs: None)

    def malformed(path: Path) -> Any:
        raise Exception("parser-specific corrupt PDF error")

    monkeypatch.setattr(collected_outcomes, "inspect_pdf", malformed)
    with fia_client() as client:
        state = scheduler_tick(tmp_path, http_client=client, now=lambda: OBSERVED)
    assert state["events"][EVENT.partition()]["outcome_collection"]["status"] == "deferred"


def test_capture_ids_win_over_a_gold_alias_for_the_same_driver() -> None:
    from f1_ml_predictor.trust.collected_outcomes import capture_driver_aliases

    lindblad = {"driverId": "arvid_lindblad", "givenName": "Arvid", "familyName": "Lindblad"}
    entries = [{"Driver": lindblad}]
    aliases = capture_driver_aliases(entries, {"arvid_lindblad"})
    assert aliases["Arvid LINDBLAD"] == "arvid_lindblad"
    # The alias target racing in the same capture under its own ID is a real conflict.
    with pytest.raises(ValueError, match="conflicts with exact FIA alias"):
        capture_driver_aliases(entries, {"arvid_lindblad", "lindblad"})
