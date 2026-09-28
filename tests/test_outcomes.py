from datetime import UTC, datetime

import pyarrow as pa
import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.outcomes import (
    DNF_TAXONOMY_VERSION,
    OUTCOME_SCHEMA,
    DnfCategory,
    audited_dnf,
    validate_audited_outcomes,
)


def outcome(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2025,
        "round": 1,
        "driver_id": "verstappen",
        "position": 1,
        "classified": True,
        "winner": True,
        "podium": True,
        "dnf": False,
        "dnf_category": "finished",
        "raw_status": "Finished",
        "taxonomy_version": DNF_TAXONOMY_VERSION,
        "final_audited": True,
        "label_available_at": datetime(2025, 3, 17, tzinfo=UTC),
        "audit_reference": "audit:official-final-2025-01",
    }
    row.update(changes)
    return row


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (category, True)
        for category in (
            DnfCategory.RETIRED_MECHANICAL,
            DnfCategory.RETIRED_INCIDENT,
            DnfCategory.RETIRED_OTHER,
        )
    ]
    + [
        (DnfCategory.FINISHED, False),
        (DnfCategory.DID_NOT_START, None),
        (DnfCategory.DISQUALIFIED, None),
        (DnfCategory.UNKNOWN, None),
    ],
)
def test_explicit_audited_taxonomy(category: DnfCategory, expected: bool | None) -> None:
    assert audited_dnf(category) is expected


@pytest.mark.parametrize("status", ["Engine", "+1 Lap", "Accident", "Finished"])
def test_provider_status_is_not_an_audited_category(status: str) -> None:
    with pytest.raises(ValueError):
        audited_dnf(status)


def test_classified_retirement_remains_dnf() -> None:
    table = pa.Table.from_pylist(
        [
            outcome(
                position=15,
                winner=False,
                podium=False,
                dnf=True,
                dnf_category="retired_mechanical",
                raw_status="+1 Lap",
            )
        ],
        schema=OUTCOME_SCHEMA,
    )
    validate_audited_outcomes(table)
    assert table["classified"][0].as_py() is True
    assert table["dnf"][0].as_py() is True


@pytest.mark.parametrize("category", ["did_not_start", "disqualified", "unknown"])
def test_dns_dsq_unknown_have_missing_dnf(category: str) -> None:
    row = outcome(
        position=None, classified=False, winner=False, podium=False, dnf=None, dnf_category=category
    )
    validate_audited_outcomes(pa.Table.from_pylist([row], schema=OUTCOME_SCHEMA))
    row["dnf"] = True
    with pytest.raises(ValueError, match="contradicts"):
        validate_audited_outcomes(pa.Table.from_pylist([row], schema=OUTCOME_SCHEMA))


@pytest.mark.parametrize(
    "changes",
    [
        {"driver_id": "invalid id"},
        {"winner": False},
        {"podium": False},
        {"final_audited": False},
        {"audit_reference": " "},
        {"taxonomy_version": "provider-v0"},
        {"label_available_at": None},
        {"position": 0},
        {"dnf_category": "Engine"},
    ],
)
def test_invalid_final_labels_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_audited_outcomes(pa.Table.from_pylist([outcome(**changes)], schema=OUTCOME_SCHEMA))


def test_duplicate_driver_and_position_rejected() -> None:
    for second in [outcome(), outcome(driver_id="hamilton")]:
        with pytest.raises(ValueError, match="duplicate|unique"):
            validate_audited_outcomes(
                pa.Table.from_pylist([outcome(), second], schema=OUTCOME_SCHEMA)
            )


def test_optional_field_roster_and_post_prediction_label_availability() -> None:
    table = pa.Table.from_pylist([outcome()], schema=OUTCOME_SCHEMA)
    validate_audited_outcomes(table, field_roster=[(EventId(2025, 1), "verstappen")])
    # Final outcomes are future labels, not inputs available before the race.
    assert table["label_available_at"][0].as_py() > datetime(2025, 3, 15, tzinfo=UTC)
    with pytest.raises(ValueError, match="roster"):
        validate_audited_outcomes(table, field_roster=[(EventId(2025, 1), "hamilton")])
