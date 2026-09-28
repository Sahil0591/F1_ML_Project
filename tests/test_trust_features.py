from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from f1_ml_predictor.features.contracts import (
    FeatureInputs,
    PreRaceEvent,
    PublishedTable,
    ResultVersion,
)
from f1_ml_predictor.features.snapshot import build_snapshot
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.cutoffs import CutoffKind
from f1_ml_predictor.trust.evidence import (
    AvailabilityEvidence,
    BenchmarkTier,
    EvidenceClass,
    table_hash,
)
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION


def dt(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2025, 3, day, hour, minute, tzinfo=UTC)


EVENT_ID = EventId(2025, 3)
CUTOFF = dt(15, 10)
EVENT_AVAILABLE = dt(15, 9)


def pub(
    rows: list[dict[str, object]],
    available: datetime,
    reference: str,
    evidence: AvailabilityEvidence | None = None,
) -> PublishedTable:
    table = pa.Table.from_pylist(rows)
    return PublishedTable(table, available, reference, evidence)


def certified_pub(
    rows: list[dict[str, object]],
    available: datetime,
    reference: str,
    *,
    kind: EvidenceClass = EvidenceClass.CAPTURED_LIVE,
    audited: bool = False,
) -> PublishedTable:
    table = pa.Table.from_pylist(rows)
    evidence = AvailabilityEvidence(
        kind=kind,
        reference=reference,
        available_at=available,
        captured_at=available if kind == EvidenceClass.CAPTURED_LIVE else None,
        artifact_sha256=table_hash(table),
        source_published_at=(
            available if kind == EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP else None
        ),
        archive_version="archive-v1" if kind == EvidenceClass.VERSIONED_ARCHIVE else None,
        reconstruction_method=(
            "conservative-v1" if kind == EvidenceClass.CONSERVATIVE_RECONSTRUCTION else None
        ),
        audited=audited,
    )
    return PublishedTable(table, available, reference, evidence)


def event_spec(
    *,
    status: str = "completed",
    available: datetime = EVENT_AVAILABLE,
    cancelled_at: datetime | None = None,
) -> PreRaceEvent:
    event = PreRaceEvent(
        EVENT_ID,
        "silverstone",
        dt(15, 14),
        None if status == "cancelled" else dt(15, 9),
        available,
        "event-proof",
        qualifying_status=status,
        qualifying_cancelled_at=cancelled_at,
    )
    evidence = AvailabilityEvidence(
        EvidenceClass.CAPTURED_LIVE,
        "event-proof",
        available,
        available,
        table_hash(event.as_table()),
    )
    return replace(event, evidence=evidence)


def roster_rows(
    *, start_type: str = "grid", grid_position: int | None = 2, grid_status: str = "final"
) -> list[dict[str, object]]:
    return [
        {
            "event_id": EVENT_ID.partition(),
            "driver_id": "driver_a",
            "constructor_id": "team_x",
            "grid_position": grid_position,
            "start_type": start_type,
            "grid_status": grid_status,
        },
        {
            "event_id": EVENT_ID.partition(),
            "driver_id": "driver_b",
            "constructor_id": "team_x",
            "grid_position": 4,
            "start_type": "grid",
            "grid_status": grid_status,
        },
    ]


def qualifying_rows() -> list[dict[str, object]]:
    return [
        {
            "event_id": EVENT_ID.partition(),
            "driver_id": driver,
            "constructor_id": "team_x",
            "position": position,
            "q3_seconds": 90.0 + position,
        }
        for driver, position in (("driver_a", 2), ("driver_b", 4))
    ]


def inputs(
    *,
    event: PreRaceEvent | None = None,
    certified: bool = False,
    sessions: tuple[PublishedTable, ...] = (),
    qualifying: tuple[PublishedTable, ...] | None = None,
    rosters: tuple[PublishedTable, ...] | None = None,
    history: tuple[ResultVersion, ...] = (),
) -> FeatureInputs:
    make_pub = certified_pub if certified else pub
    roster = make_pub(roster_rows(), dt(15, 9, 30), "roster-proof")
    q = make_pub(qualifying_rows(), dt(15, 9, 20), "qualifying-proof")
    return FeatureInputs(
        event=event or event_spec(),
        rosters=rosters if rosters is not None else (roster,),
        qualifying=qualifying if qualifying is not None else (q,),
        history=history,
        sessions=sessions,
    )


def test_evidence_classes_do_not_promote_without_required_audit_fields() -> None:
    table = pa.table({"value": [1]})
    evidence = AvailabilityEvidence(
        EvidenceClass.VERSIONED_ARCHIVE,
        "archive",
        dt(15, 9),
        artifact_sha256=table_hash(table),
        archive_version="v1",
    )
    assert evidence.tier == BenchmarkTier.DEVELOPMENT
    audited_reconstruction = replace(
        evidence,
        kind=EvidenceClass.CONSERVATIVE_RECONSTRUCTION,
        archive_version=None,
        reconstruction_method="method-v1",
        audited=True,
    )
    assert audited_reconstruction.tier == BenchmarkTier.SILVER
    captured = AvailabilityEvidence(
        EvidenceClass.CAPTURED_LIVE,
        "capture",
        dt(15, 9),
        dt(15, 9),
        table_hash(table),
    )
    assert captured.tier == BenchmarkTier.GOLD
    with pytest.raises(ValueError, match="capture time"):
        AvailabilityEvidence(
            EvidenceClass.CAPTURED_LIVE,
            "bad-capture",
            dt(15, 9),
            dt(15, 8, 59),
            table_hash(table),
        )


@pytest.mark.parametrize(
    ("kind", "archive_version", "reconstruction_method"),
    [
        (EvidenceClass.VERSIONED_ARCHIVE, "archive-v1", None),
        (EvidenceClass.CONSERVATIVE_RECONSTRUCTION, None, "reconstruction-v1"),
    ],
)
def test_archive_evidence_cannot_claim_source_publication_after_availability(
    kind: EvidenceClass,
    archive_version: str | None,
    reconstruction_method: str | None,
) -> None:
    table = pa.table({"value": [1]})
    with pytest.raises(ValueError):
        AvailabilityEvidence(
            kind=kind,
            reference="contradictory-release-proof",
            available_at=dt(15, 9),
            artifact_sha256=table_hash(table),
            source_published_at=dt(15, 10),
            archive_version=archive_version,
            reconstruction_method=reconstruction_method,
            audited=True,
        )


def test_event_evidence_cannot_be_available_before_qualifying_completion() -> None:
    event = PreRaceEvent(
        EVENT_ID,
        "silverstone",
        dt(15, 14),
        dt(15, 9),
        dt(15, 8, 59),
        "early-event-capture",
    )
    evidence = AvailabilityEvidence(
        EvidenceClass.CAPTURED_LIVE,
        "early-event-capture",
        event.available_at,
        event.available_at,
        table_hash(event.as_table()),
    )
    with pytest.raises(ValueError, match="available"):
        replace(event, evidence=evidence)


def test_table_evidence_binds_exact_hash_and_legacy_publication_is_development() -> None:
    table = pa.table({"value": [1]})
    evidence = AvailabilityEvidence(
        EvidenceClass.CAPTURED_LIVE,
        "captured",
        dt(15, 9),
        dt(15, 9),
        table_hash(table),
    )
    assert PublishedTable(table, dt(15, 9), "captured", evidence).tier == BenchmarkTier.GOLD
    with pytest.raises(ValueError, match="exact table version"):
        PublishedTable(pa.table({"value": [2]}), dt(15, 9), "captured", evidence)
    assert pub([{"value": 1}], dt(15, 9), "legacy").tier == BenchmarkTier.DEVELOPMENT


def test_certified_only_rejects_legacy_required_development_input() -> None:
    with pytest.raises(ValueError, match="latest required"):
        build_snapshot(inputs(), CUTOFF, certified_only=True)


def test_newer_development_roster_cannot_fall_back_to_older_gold_version() -> None:
    older_gold = certified_pub(roster_rows(), dt(15, 9, 10), "gold-roster")
    newer_development = pub(roster_rows(grid_position=10), dt(15, 9, 30), "development-roster")
    with pytest.raises(ValueError, match="latest required publication"):
        build_snapshot(
            inputs(
                certified=True,
                rosters=(older_gold, newer_development),
            ),
            CUTOFF,
            certified_only=True,
        )


def test_gold_captured_inputs_and_silver_audited_reconstruction_are_reported() -> None:
    gold = build_snapshot(inputs(certified=True), CUTOFF)
    assert set(gold.column("benchmark_tier").to_pylist()) == {"Gold"}

    roster = certified_pub(
        roster_rows(),
        dt(15, 9, 30),
        "roster-reconstructed",
        kind=EvidenceClass.CONSERVATIVE_RECONSTRUCTION,
        audited=True,
    )
    silver = build_snapshot(inputs(certified=True, rosters=(roster,)), CUTOFF)
    assert set(silver.column("benchmark_tier").to_pylist()) == {"Silver"}


def test_certified_only_preserves_missing_optional_feature_when_input_is_development() -> None:
    session = pub(
        [
            {
                "event_id": EVENT_ID.partition(),
                "driver_id": "driver_a",
                "session_code": "FP2",
                "source": "fastf1",
                "best_lap_seconds": 89.0,
            }
        ],
        dt(15, 9, 45),
        "legacy-practice",
    )
    snapshot = build_snapshot(
        inputs(certified=True, sessions=(session,)), CUTOFF, certified_only=True
    )
    row = snapshot.to_pylist()[0]
    assert row["practice_best_seconds"] is None
    assert row["practice_best_seconds_missing"] is True
    provenance = json.loads(row["provenance"])
    assert {entry["reference"] for entry in provenance["excluded_optional"]} == {"legacy-practice"}


def test_newer_development_forecast_correction_cannot_fall_back_to_older_gold() -> None:
    def forecast(reference: str, available: datetime, temperature: float, gold: bool):
        return (certified_pub if gold else pub)(
            [
                {
                    "event_id": EVENT_ID.partition(),
                    "weather_kind": "forecast",
                    "valid_at": dt(15, 14),
                    "captured_at": available,
                    "available_at": available,
                    "temperature_2m": temperature,
                }
            ],
            available,
            reference,
        )

    older_gold = forecast("gold-weather", dt(15, 9, 10), 18.0, True)
    newer_development = forecast("development-weather-correction", dt(15, 9, 30), 21.0, False)
    snapshot = build_snapshot(
        replace(
            inputs(certified=True),
            forecasts=(older_gold, newer_development),
        ),
        CUTOFF,
        certified_only=True,
    )
    row = snapshot.to_pylist()[0]
    assert row["forecast_temperature_2m"] is None
    assert row["forecast_temperature_2m_missing"] is True
    provenance = json.loads(row["provenance"])
    assert {item["reference"] for item in provenance["excluded_optional"]} == {
        "development-weather-correction"
    }


def test_newer_development_history_correction_cannot_fall_back_to_older_gold() -> None:
    previous = EventId(2025, 2)
    older_gold = history_result(audited=True)
    corrected_row = {
        "event_id": previous.partition(),
        "driver_id": "driver_a",
        "constructor_id": "team_x",
        "position": 8,
        "dnf": False,
        "pit_stop_seconds": 24.0,
    }
    newer_development = ResultVersion(
        previous,
        dt(14, 15),
        pub([corrected_row], dt(15, 9, 30), "development-history-correction"),
    )
    snapshot = build_snapshot(
        inputs(certified=True, history=(older_gold, newer_development)),
        CUTOFF,
        certified_only=True,
    )
    row = snapshot.to_pylist()[0]
    assert row["history_count"] == 0
    assert row["recent_finish_mean"] is None
    assert row["recent_dnf_rate"] is None
    provenance = json.loads(row["provenance"])
    assert {item["reference"] for item in provenance["excluded_optional"]} == {
        "development-history-correction"
    }


def test_cancelled_qualifying_requires_published_cancellation_and_allows_no_classification() -> (
    None
):
    cancelled_event = event_spec(
        status="cancelled", available=dt(15, 9, 35), cancelled_at=dt(15, 9, 30)
    )
    roster = certified_pub(roster_rows(), dt(15, 9, 40), "roster-proof")
    snapshot = build_snapshot(
        inputs(event=cancelled_event, certified=True, qualifying=(), rosters=(roster,)), CUTOFF
    )
    assert set(snapshot.column("qualifying_status").to_pylist()) == {"cancelled"}
    assert snapshot.to_pylist()[0]["qualifying_position"] is None

    with pytest.raises(ValueError, match="timestamp"):
        event_spec(status="cancelled")


def test_cancelled_event_omits_empty_development_qualifying_response() -> None:
    cancelled_event = event_spec(
        status="cancelled", available=dt(15, 9, 35), cancelled_at=dt(15, 9, 30)
    )
    roster = certified_pub(roster_rows(), dt(15, 9, 40), "roster-proof")
    empty_qualifying = pub([], dt(15, 9, 41), "uncertified-empty-qualifying")
    row = build_snapshot(
        inputs(
            event=cancelled_event,
            certified=True,
            qualifying=(empty_qualifying,),
            rosters=(roster,),
        ),
        CUTOFF,
        certified_only=True,
    ).to_pylist()[0]
    assert row["qualifying_status"] == "cancelled"
    assert row["qualifying_position"] is None
    assert "uncertified-empty-qualifying" in {
        item["reference"] for item in json.loads(row["provenance"])["excluded_optional"]
    }


def test_live_evidence_rejects_rows_captured_after_outer_capture_time() -> None:
    table = pa.Table.from_pylist([{"available_at": dt(15, 9), "captured_at": dt(15, 12)}])
    evidence = AvailabilityEvidence(
        EvidenceClass.CAPTURED_LIVE,
        "capture-proof",
        dt(15, 9),
        dt(15, 9),
        table_hash(table),
    )
    with pytest.raises(ValueError, match="after prediction_timestamp"):
        PublishedTable(table, dt(15, 9), "capture-proof", evidence)


def test_pit_lane_start_has_no_grid_ordinal_and_provisional_grid_requires_status() -> None:
    roster = certified_pub(
        roster_rows(start_type="pit_lane", grid_position=None, grid_status="provisional"),
        dt(15, 9, 30),
        "provisional-roster",
    )
    table = build_snapshot(
        inputs(certified=True, rosters=(roster,)),
        dt(15, 9, 40),
        cutoff_kind=CutoffKind.PROVISIONAL_GRID,
    )
    row = table.to_pylist()[0]
    assert row["pit_lane_start"] is True
    assert row["grid_position"] is None
    assert row["grid_status"] == "provisional"

    unknown = certified_pub(roster_rows(grid_status="unknown"), dt(15, 9, 30), "unknown-grid")
    with pytest.raises(ValueError, match="published grid status"):
        build_snapshot(
            inputs(certified=True, rosters=(unknown,)),
            dt(15, 9, 40),
            cutoff_kind=CutoffKind.PROVISIONAL_GRID,
        )


def test_pre_race_cutoff_enforces_declared_window() -> None:
    with pytest.raises(ValueError, match="outside the declared pre-race window"):
        build_snapshot(inputs(certified=True), dt(15, 12), cutoff_kind=CutoffKind.PRE_RACE)
    table = build_snapshot(inputs(certified=True), dt(15, 13, 1), cutoff_kind=CutoffKind.PRE_RACE)
    assert set(table.column("cutoff_kind").to_pylist()) == {"pre_race"}


def session_pub(source: str, seconds: float) -> PublishedTable:
    return certified_pub(
        [
            {
                "event_id": EVENT_ID.partition(),
                "driver_id": "driver_a",
                "session_code": "FP2",
                "source": source,
                "best_lap_seconds": seconds,
            }
        ],
        dt(15, 9, 45),
        f"{source}-practice",
    )


def test_provider_disagreement_is_visible_not_averaged_and_quarantined_when_certified() -> None:
    sessions = (session_pub("fastf1", 90.0), session_pub("openf1", 80.0))
    inputs_value = inputs(certified=True, sessions=sessions)
    development = build_snapshot(inputs_value, CUTOFF)
    row = development.to_pylist()[0]
    assert row["practice_best_seconds"] == 90.0
    assert row["benchmark_tier"] == "Development"
    report = json.loads(row["provenance"])["arbitration"][0]
    assert report["disagreement"] is True
    assert report["policy"] == "evidence_then_preference_no_average"

    certified = build_snapshot(inputs_value, CUTOFF, certified_only=True).to_pylist()[0]
    assert certified["practice_best_seconds"] is None
    assert certified["practice_best_seconds_missing"] is True
    assert certified["benchmark_tier"] == "Gold"
    assert json.loads(certified["provenance"])["arbitration"][0]["disagreement"] is True


def test_mixed_tier_provider_disagreement_is_quarantined_without_false_gold_practice() -> None:
    gold_fastf1 = session_pub("fastf1", 90.0)
    development_openf1 = pub(
        [
            {
                "event_id": EVENT_ID.partition(),
                "driver_id": "driver_a",
                "session_code": "FP2",
                "source": "openf1",
                "best_lap_seconds": 95.0,
            }
        ],
        dt(15, 9, 46),
        "development-openf1-practice",
    )
    snapshot = build_snapshot(
        inputs(certified=True, sessions=(gold_fastf1, development_openf1)),
        CUTOFF,
        certified_only=True,
    )
    row = snapshot.to_pylist()[0]
    assert row["practice_best_seconds"] is None
    assert row["practice_best_seconds_missing"] is True
    assert row["benchmark_tier"] == "Gold"
    provenance = json.loads(row["provenance"])
    assert provenance["arbitration"][0]["disagreement"] is True
    assert provenance["arbitration"][0]["candidates"] == ["fastf1", "openf1"]
    assert provenance["arbitration"][0]["selected"] == "fastf1"


def history_result(*, audited: bool) -> ResultVersion:
    previous = EventId(2025, 2)
    row: dict[str, object] = {
        "event_id": previous.partition(),
        "driver_id": "driver_a",
        "constructor_id": "team_x",
        "position": 8,
        "dnf": True,
        "pit_stop_seconds": 24.0,
    }
    if audited:
        row.update(
            {
                "final_audited": True,
                "audit_reference": "fia-final-classification",
                "taxonomy_version": DNF_TAXONOMY_VERSION,
                "dnf_category": "retired_mechanical",
            }
        )
    publication = certified_pub([row], dt(15, 9, 10), "prior-result")
    return ResultVersion(previous, dt(14, 15), publication)


def test_legacy_dnf_is_development_and_certified_mode_uses_only_audited_taxonomy() -> None:
    legacy = build_snapshot(
        inputs(certified=True, history=(history_result(audited=False),)), CUTOFF
    )
    assert legacy.to_pylist()[0]["recent_dnf_rate"] == 1.0
    assert legacy.to_pylist()[0]["benchmark_tier"] == "Development"

    legacy_certified = build_snapshot(
        inputs(certified=True, history=(history_result(audited=False),)),
        CUTOFF,
        certified_only=True,
    )
    assert legacy_certified.to_pylist()[0]["recent_dnf_rate"] is None
    assert legacy_certified.to_pylist()[0]["benchmark_tier"] == "Gold"

    audited = build_snapshot(
        inputs(certified=True, history=(history_result(audited=True),)),
        CUTOFF,
        certified_only=True,
    )
    assert audited.to_pylist()[0]["recent_dnf_rate"] == 1.0
    assert audited.to_pylist()[0]["benchmark_tier"] == "Gold"
