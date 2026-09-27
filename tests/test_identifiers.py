import pytest

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId


def test_event_id_has_stable_partition() -> None:
    assert EventId(2026, 1).partition() == "season=2026/round=01"


@pytest.mark.parametrize("season,round_number", [(1949, 1), (2026, 0), (True, 1), (2026, False)])
def test_event_id_rejects_invalid_values(season: int, round_number: int) -> None:
    with pytest.raises(ValueError):
        EventId(season, round_number)


def test_entity_id_uses_canonical_source_id() -> None:
    assert EntityId(EntityKind.CONSTRUCTOR, "red_bull").value == "red_bull"


@pytest.mark.parametrize("value", ["", "Max Verstappen", "red-bull", "../driver", "23"])
def test_entity_id_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError):
        EntityId(EntityKind.DRIVER, value)
