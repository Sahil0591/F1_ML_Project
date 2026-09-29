"""Offline engineering fixtures, not measurements of F1 predictive accuracy."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.models.distributions import sample_race_orders
from f1_ml_predictor.simulation import (
    CurrentStandings,
    EventSimulation,
    PointsRules,
    ValidationEvidence,
    simulate_championship,
)

CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)
HASH = "a" * 64
ROSTER = {"driver_a": "team_a", "driver_b": "team_a", "driver_c": "team_b"}
ORDER = ("driver_a", "driver_b", "driver_c")
FIA_2025 = (
    "https://www.fia.com/system/files/documents/"
    "fia_2025_formula_1_sporting_regulations_-_issue_5_-_2025-04-30.pdf"
)


def rules(**changes: object) -> PointsRules:
    return replace(
        PointsRules(2025, "race", (25, 18, 15, 12, 10, 8, 6, 4, 2, 1), "2025-full", FIA_2025),
        **changes,
    )


def standings(**changes: object) -> CurrentStandings:
    return replace(
        CurrentStandings(2025, CUTOFF, dict.fromkeys(ROSTER, 0), {"team_a": 0, "team_b": 0}, HASH),
        **changes,
    )


def event(**changes: object) -> EventSimulation:
    return replace(
        EventSimulation(
            event_id=EventId(2025, 10),
            scheduled_at=CUTOFF + timedelta(days=7),
            available_at=CUTOFF,
            driver_constructors=ROSTER,
            sampled_orders=(ORDER,),
            points_eligible_samples=(ORDER,),
            rules=rules(),
            source_hash=HASH,
            model_id="test_model",
        ),
        **changes,
    )


def evidence(**changes: object) -> ValidationEvidence:
    return replace(
        ValidationEvidence(
            model_id="test_model",
            model_validated=True,
            evidence_tier="Gold",
            independent_gold_events=tuple(EventId(2024, round_) for round_ in range(1, 25))
            + (EventId(2025, 1),),
            available_at=CUTOFF,
            source_hash=HASH,
        ),
        **changes,
    )


def test_exact_scores_are_added_to_known_points_for_both_titles() -> None:
    result = simulate_championship(
        standings(driver_points={"driver_a": 0, "driver_b": 8, "driver_c": 20}),
        [event()],
        prediction_timestamp=CUTOFF,
        simulations=31,
    )
    assert result.wdc.mean_final_points == {"driver_a": 25, "driver_b": 26, "driver_c": 35}
    assert result.wdc.title_probability == {"driver_a": 0, "driver_b": 0, "driver_c": 1}
    assert result.wcc.mean_final_points == {"team_a": 43, "team_b": 15}
    assert result.wcc.title_probability == {"team_a": 1, "team_b": 0}
    assert result.status == "engineering_only"


def test_resamples_complete_orders_preserving_impossible_marginal_combinations() -> None:
    # Only a or c can win, while b is always second. Independent marginal
    # draws could invent a and c both winning the same event, changing both titles.
    joint = event(
        sampled_orders=(ORDER, tuple(reversed(ORDER))),
        points_eligible_samples=(ORDER, ORDER),
        rules=rules(points_by_position=(1, 0, 0)),
    )
    result = simulate_championship(
        standings(), [joint], prediction_timestamp=CUTOFF, simulations=10000
    )
    assert result.wdc.title_probability["driver_b"] == 0
    assert 0.47 < result.wdc.title_probability["driver_a"] < 0.53
    assert result.wdc.title_probability["driver_a"] == result.wcc.title_probability["team_a"]
    assert sum(result.wdc.title_probability.values()) == 1
    assert sum(result.wcc.title_probability.values()) == 1
    assert result.wdc.unresolved_tie_probability == 0


def test_consumes_the_shared_race_joint_sampler_without_redrawing_marginals() -> None:
    index_samples = sample_race_orders([0.1, 0.3, 0.5], [0.02, 0.1, 0.2], draws=128, seed=15)
    joint_orders = tuple(tuple(ORDER[index] for index in sample) for sample in index_samples)
    joint = event(
        sampled_orders=joint_orders,
        points_eligible_samples=tuple(ORDER for _ in joint_orders),
        eligibility_policy="all_entered_engineering_assumption",
    )
    result = simulate_championship(
        standings(), [joint], prediction_timestamp=CUTOFF, simulations=1000, seed=15
    )
    assert result.status == "engineering_only"
    assert sum(result.wdc.title_probability.values()) == pytest.approx(1)
    assert sum(result.wdc.mean_final_points.values()) == pytest.approx(58)
    assert sum(result.wcc.mean_final_points.values()) == pytest.approx(58)


def test_repeated_joint_orders_preserve_their_empirical_probability_mass() -> None:
    joint = event(
        sampled_orders=(ORDER, ORDER, ORDER, tuple(reversed(ORDER))),
        points_eligible_samples=(ORDER, ORDER, ORDER, ORDER),
        rules=rules(points_by_position=(1, 0, 0)),
    )
    result = simulate_championship(
        standings(), [joint], prediction_timestamp=CUTOFF, simulations=10000, seed=27
    )
    assert result.wdc.title_probability["driver_a"] == pytest.approx(0.75, abs=0.025)
    assert result.wdc.title_probability["driver_c"] == pytest.approx(0.25, abs=0.025)


def test_seed_hash_serialization_and_event_order_are_repeatable(tmp_path: Path) -> None:
    first = event(
        sampled_orders=(ORDER, tuple(reversed(ORDER))), points_eligible_samples=(ORDER, ORDER)
    )
    second = replace(first, event_id=EventId(2025, 11), scheduled_at=CUTOFF + timedelta(days=14))
    result = simulate_championship(
        standings(), [first, second], prediction_timestamp=CUTOFF, simulations=257, seed=19
    )
    again = simulate_championship(
        standings(), [second, first], prediction_timestamp=CUTOFF, simulations=257, seed=19
    )
    assert result.to_dict() == again.to_dict()
    changed = simulate_championship(
        standings(), [first, second], prediction_timestamp=CUTOFF, simulations=257, seed=20
    )
    assert result.to_dict()["sha256"] != changed.to_dict()["sha256"]
    path = tmp_path / "nested" / "championship.json"
    result.write_json(path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == result.to_dict()
    digest = stored.pop("sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(stored, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def test_unresolved_ties_keep_probability_mass_and_fractional_points_exact() -> None:
    # Exact rational arithmetic avoids a false winner from 0.1 + 0.2 != 0.3.
    current = standings(
        driver_points={"driver_a": 0.1, "driver_b": 0.3, "driver_c": 0},
        constructor_points={"team_a": -0.2, "team_b": 0},
    )
    shortened = event(
        rules=rules(points_by_position=(0.2, 0, 0), distance_regime="explicit_shortened"),
    )
    result = simulate_championship(current, [shortened], prediction_timestamp=CUTOFF, simulations=7)
    assert sum(result.wdc.title_probability.values()) == 0
    assert result.wdc.unresolved_tie_probability == 1
    assert result.wdc.tied_for_title_probability == {"driver_a": 1, "driver_b": 1, "driver_c": 0}
    assert result.wcc.unresolved_tie_probability == 1
    assert sum(result.wcc.title_probability.values()) + result.wcc.unresolved_tie_probability == 1


def test_constructor_assignment_is_event_specific_without_reallocating_old_points() -> None:
    transferred = event(
        driver_constructors={"driver_a": "team_b", "driver_b": "team_a", "driver_c": "team_b"}
    )
    result = simulate_championship(
        standings(constructor_points={"team_a": 100, "team_b": 0}),
        [transferred],
        prediction_timestamp=CUTOFF,
        simulations=1,
    )
    assert result.wcc.mean_final_points == {"team_a": 118, "team_b": 40}


def test_ineligible_drivers_receive_no_points_and_are_not_promoted_into_the_table() -> None:
    explicit = event(points_eligible_samples=(("driver_a", "driver_c"),))
    result = simulate_championship(
        standings(), [explicit], prediction_timestamp=CUTOFF, simulations=1
    )
    assert result.wdc.mean_final_points == {"driver_a": 25, "driver_b": 0, "driver_c": 15}
    assert result.wcc.mean_final_points == {"team_a": 25, "team_b": 15}


def test_race_and_sprint_can_share_round_but_not_same_event_kind() -> None:
    sprint = event(
        rules=rules(event_kind="sprint", points_by_position=(8, 7, 6, 5, 4, 3, 2, 1)),
        scheduled_at=CUTOFF + timedelta(days=6),
    )
    result = simulate_championship(
        standings(), [event(), sprint], prediction_timestamp=CUTOFF, simulations=1
    )
    assert result.wdc.mean_final_points == {"driver_a": 33, "driver_b": 25, "driver_c": 21}
    with pytest.raises(ValueError, match="duplicate"):
        simulate_championship(standings(), [event(), event()], prediction_timestamp=CUTOFF)


def test_2024_fastest_lap_is_explicit_and_2025_bonus_is_rejected() -> None:
    old_rules = rules(season=2024, fastest_lap_points=1, fastest_lap_max_position=10)
    old_event = event(
        event_id=EventId(2024, 10), rules=old_rules, fastest_lap_drivers=("driver_c",)
    )
    result = simulate_championship(
        standings(season=2024), [old_event], prediction_timestamp=CUTOFF, simulations=1
    )
    assert result.wdc.mean_final_points["driver_c"] == 16
    assert result.wcc.mean_final_points["team_b"] == 16
    with pytest.raises(ValueError, match="2025"):
        rules(fastest_lap_points=1, fastest_lap_max_position=10)
    with pytest.raises(ValueError, match="one explicit driver"):
        event(event_id=EventId(2024, 10), rules=old_rules)
    with pytest.raises(ValueError, match="no bonus"):
        event(fastest_lap_drivers=("driver_c",))


def test_fastest_lap_outside_top_ten_or_ineligible_does_not_score() -> None:
    identifiers = tuple(f"driver_{index}" for index in range(11))
    roster = {driver: f"team_{index // 2}" for index, driver in enumerate(identifiers)}
    current = CurrentStandings(
        2024, CUTOFF, dict.fromkeys(roster, 0), dict.fromkeys(roster.values(), 0), HASH
    )
    old_rules = rules(season=2024, fastest_lap_points=1, fastest_lap_max_position=10)
    old_event = EventSimulation(
        EventId(2024, 10),
        CUTOFF + timedelta(days=1),
        CUTOFF,
        roster,
        (identifiers,),
        (identifiers,),
        old_rules,
        HASH,
        fastest_lap_drivers=(identifiers[-1],),
    )
    result = simulate_championship(current, [old_event], prediction_timestamp=CUTOFF, simulations=1)
    assert result.wdc.mean_final_points[identifiers[-1]] == 0
    ineligible = replace(
        old_event, points_eligible_samples=(identifiers[1:],), fastest_lap_drivers=(identifiers[0],)
    )
    result = simulate_championship(
        current, [ineligible], prediction_timestamp=CUTOFF, simulations=1
    )
    assert result.wdc.mean_final_points[identifiers[0]] == 0


@pytest.mark.parametrize(
    "change, message",
    [
        ({"sampled_orders": (("driver_a", "driver_a", "driver_c"),)}, "complete permutation"),
        ({"sampled_orders": (("driver_a", "driver_b"),)}, "complete permutation"),
        ({"points_eligible_samples": ()}, "corresponding"),
        ({"points_eligible_samples": (("driver_a", "absent"),)}, "unique subset"),
        ({"points_eligible_samples": (("driver_a", "driver_a"),)}, "unique subset"),
        ({"driver_constructors": {"NOT_CANONICAL": "team_a"}}, "canonical Jolpica"),
        ({"source_hash": "not a hash"}, "SHA-256"),
        ({"eligibility_policy": "guess_from_dnf"}, "must be explicit"),
    ],
)
def test_rejects_incoherent_samples_and_uncanonical_sources(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        event(**change)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"season": 1950}, "supported"),
        ({"season": True}, "supported"),
        ({"season": 2025.0}, "supported"),
        ({"distance_regime": "rain"}, "explicitly select"),
        ({"event_kind": "qualifying"}, "race or sprint"),
        ({"points_by_position": ()}, "explicit nonempty"),
        ({"points_by_position": (1, 2)}, "nonincreasing"),
        ({"points_by_position": (float("nan"),)}, "finite"),
        ({"points_by_position": (-1,)}, "nonnegative"),
        ({"distance_regime": "no_points"}, "zero position"),
        ({"season": 2020, "event_kind": "sprint"}, "before 2021"),
    ],
)
def test_rejects_ambiguous_or_unsupported_rules(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rules(**change)


def test_no_points_event_and_empty_remaining_calendar_preserve_standings() -> None:
    current = standings(driver_points={"driver_a": 1, "driver_b": 0, "driver_c": 0})
    no_points = event(rules=rules(points_by_position=(0,), distance_regime="no_points"))
    zero_result = simulate_championship(
        current, [no_points], prediction_timestamp=CUTOFF, simulations=3
    )
    empty_result = simulate_championship(current, [], prediction_timestamp=CUTOFF, simulations=3)
    assert zero_result.wdc == empty_result.wdc
    assert zero_result.wcc == empty_result.wcc
    assert empty_result.wdc.title_probability["driver_a"] == 1


def test_cutoff_rejects_late_standings_predictions_and_validation() -> None:
    late = CUTOFF + timedelta(seconds=1)
    with pytest.raises(ValueError, match="after prediction"):
        simulate_championship(standings(available_at=late), [], prediction_timestamp=CUTOFF)
    with pytest.raises(ValueError, match="after prediction"):
        simulate_championship(standings(), [event(available_at=late)], prediction_timestamp=CUTOFF)
    with pytest.raises(ValueError, match="after prediction"):
        simulate_championship(
            standings(),
            [event()],
            prediction_timestamp=CUTOFF,
            validation=evidence(available_at=late),
        )
    with pytest.raises(ValueError, match="after prediction_timestamp"):
        simulate_championship(
            standings(), [event(scheduled_at=CUTOFF)], prediction_timestamp=CUTOFF
        )
    with pytest.raises(ValueError, match="UTC"):
        standings(available_at=CUTOFF.replace(tzinfo=None))


def test_missing_zero_point_driver_and_constructor_are_not_silently_invented() -> None:
    with pytest.raises(ValueError, match="all remaining drivers"):
        simulate_championship(
            standings(driver_points={"driver_a": 0}), [event()], prediction_timestamp=CUTOFF
        )
    with pytest.raises(ValueError, match="all remaining constructors"):
        simulate_championship(
            standings(constructor_points={"team_a": 0}), [event()], prediction_timestamp=CUTOFF
        )


def test_fixture_status_cannot_be_promoted_by_flag_without_25_gold_races() -> None:
    for invalid in (
        evidence(evidence_tier="Silver"),
        evidence(independent_gold_events=tuple(EventId(2024, n) for n in range(1, 25))),
    ):
        with pytest.raises(ValueError, match="25 independent Gold"):
            simulate_championship(
                standings(), [event()], prediction_timestamp=CUTOFF, validation=invalid
            )
    for invalid, message in (
        (event(model_id="other_model"), "model_id"),
        (event(event_id=EventId(2025, 1)), "independent"),
        (event(eligibility_policy="all_entered_engineering_assumption"), "explicit modeled"),
    ):
        with pytest.raises(ValueError, match=message):
            simulate_championship(
                standings(), [invalid], prediction_timestamp=CUTOFF, validation=evidence()
            )
    result = simulate_championship(
        standings(), [event()], prediction_timestamp=CUTOFF, simulations=1, validation=evidence()
    )
    assert result.status == "validated_model_scenario"
    assert result.provenance["validation"] == evidence().to_dict()
    exploratory = simulate_championship(
        standings(),
        [event()],
        prediction_timestamp=CUTOFF,
        simulations=1,
        validation=evidence(model_validated=False, evidence_tier="Development"),
    )
    assert exploratory.status == "engineering_only"


def test_hash_binds_classification_scoring_and_current_points() -> None:
    baseline = simulate_championship(
        standings(), [event()], prediction_timestamp=CUTOFF, simulations=1
    )
    variants = (
        simulate_championship(
            standings(),
            [event(points_eligible_samples=(("driver_a",),))],
            prediction_timestamp=CUTOFF,
            simulations=1,
        ),
        simulate_championship(
            standings(),
            [event(rules=rules(points_by_position=(1,)))],
            prediction_timestamp=CUTOFF,
            simulations=1,
        ),
        simulate_championship(
            standings(driver_points={"driver_a": 1, "driver_b": 0, "driver_c": 0}),
            [event()],
            prediction_timestamp=CUTOFF,
            simulations=1,
        ),
    )
    assert all(
        variant.provenance["input_sha256"] != baseline.provenance["input_sha256"]
        for variant in variants
    )


@pytest.mark.parametrize("future", [EventId(2026, 1), EventId(2025, 11), EventId(2025, 20)])
def test_validation_rejects_future_races_even_when_claimed_available(future: EventId) -> None:
    claimed = evidence(
        independent_gold_events=tuple(EventId(2024, n) for n in range(1, 25)) + (future,)
    )
    with pytest.raises(ValueError, match="future season or later round"):
        simulate_championship(
            standings(), [event()], prediction_timestamp=CUTOFF, validation=claimed
        )


def test_preseason_forecast_cannot_use_next_season_validation_races() -> None:
    future_event = event(
        event_id=EventId(2026, 10),
        scheduled_at=datetime(2026, 6, 1, tzinfo=UTC),
        rules=rules(season=2026),
    )
    claimed = evidence(
        independent_gold_events=tuple(EventId(2024, n) for n in range(1, 25)) + (EventId(2026, 1),)
    )
    with pytest.raises(ValueError, match="future season"):
        simulate_championship(
            standings(season=2026),
            [future_event],
            prediction_timestamp=CUTOFF,
            validation=claimed,
        )
    earlier = simulate_championship(
        standings(season=2026),
        [future_event],
        prediction_timestamp=CUTOFF,
        validation=evidence(),
        simulations=1,
    )
    assert earlier.status == "validated_model_scenario"
