"""Final audited outcome labels; retirement and classification are independent."""

from collections.abc import Collection
from enum import StrEnum

import pyarrow as pa

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_utc

DNF_TAXONOMY_VERSION = "audited-dnf-v1"


class DnfCategory(StrEnum):
    FINISHED = "finished"
    RETIRED_MECHANICAL = "retired_mechanical"
    RETIRED_INCIDENT = "retired_incident"
    RETIRED_OTHER = "retired_other"
    DID_NOT_START = "did_not_start"
    DISQUALIFIED = "disqualified"
    UNKNOWN = "unknown"


def audited_dnf(category: DnfCategory | str) -> bool | None:
    """Map only the audited taxonomy, never a provider's free-form status."""
    category = DnfCategory(category)
    if category == DnfCategory.FINISHED:
        return False
    if category in {
        DnfCategory.RETIRED_MECHANICAL,
        DnfCategory.RETIRED_INCIDENT,
        DnfCategory.RETIRED_OTHER,
    }:
        return True
    return None


OUTCOME_SCHEMA = pa.schema(
    [
        pa.field("season", pa.int64(), nullable=False),
        pa.field("round", pa.int64(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("position", pa.int64()),
        pa.field("classified", pa.bool_(), nullable=False),
        pa.field("winner", pa.bool_(), nullable=False),
        pa.field("podium", pa.bool_(), nullable=False),
        pa.field("dnf", pa.bool_()),
        pa.field("dnf_category", pa.string(), nullable=False),
        pa.field("raw_status", pa.string(), nullable=False),
        pa.field("taxonomy_version", pa.string(), nullable=False),
        pa.field("final_audited", pa.bool_(), nullable=False),
        pa.field("label_available_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("audit_reference", pa.string(), nullable=False),
    ]
)


def validate_audited_outcomes(
    table: pa.Table,
    *,
    field_roster: Collection[tuple[EventId, str]] | None = None,
) -> None:
    """Validate final labels without applying a prediction-input cutoff.

    Each label retains its availability for later training cutoffs. An optional
    roster makes completeness an explicit caller decision.
    """
    for field in OUTCOME_SCHEMA:
        if field.name not in table.column_names:
            raise ValueError(f"missing outcome column: {field.name}")
        column = table[field.name]
        if column.type != field.type:
            raise ValueError(f"invalid outcome type: {field.name}")
        if not field.nullable and column.null_count:
            raise ValueError(f"missing outcome value: {field.name}")
    keys: set[tuple[EventId, str]] = set()
    positions: set[tuple[EventId, int]] = set()
    for row in table.to_pylist():
        event = EventId(row["season"], row["round"])
        EntityId(EntityKind.DRIVER, row["driver_id"])
        key = (event, row["driver_id"])
        if key in keys:
            raise ValueError("duplicate event/driver outcome")
        keys.add(key)
        if not row["final_audited"]:
            raise ValueError("outcome labels must be final audited")
        require_utc(row["label_available_at"], "label_available_at")
        if not row["audit_reference"].strip():
            raise ValueError("outcome audit reference is required")
        if row["taxonomy_version"] != DNF_TAXONOMY_VERSION:
            raise ValueError("unsupported DNF taxonomy version")
        if row["dnf"] != audited_dnf(row["dnf_category"]):
            raise ValueError("DNF label contradicts audited category")
        position = row["position"]
        if position is not None:
            if position < 1 or (event, position) in positions:
                raise ValueError("outcome positions must be positive and unique per event")
            positions.add((event, position))
        winner = position == 1 and row["classified"]
        podium = position is not None and position <= 3 and row["classified"]
        if row["winner"] != winner or row["podium"] != podium:
            raise ValueError("winner/podium labels contradict classified position")
        if row["classified"] and position is None:
            raise ValueError("classified outcomes require a position")
        if (
            row["dnf_category"]
            in {
                DnfCategory.DID_NOT_START,
                DnfCategory.DISQUALIFIED,
            }
            and row["classified"]
        ):
            raise ValueError("DNS/DSQ outcomes cannot be classified")
    if field_roster is not None and keys != set(field_roster):
        raise ValueError("outcomes do not match the supplied field roster")
