"""Monte Carlo title scenarios using whole, externally supplied joint orders.

There are no default scoring tables or draws from per-driver marginals. Each
event supplies an explicit scoring contract and points eligibility alongside
every complete order. Point ties remain unresolved because current points do
not contain the historical finish counts needed for FIA championship countback.
"""

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.models.protocol import SELECTION_PAIRED_EVENTS
from f1_ml_predictor.time import require_known_by, require_utc

_SUPPORTED_SEASONS = frozenset(range(2019, 2027))
_EVENT_KINDS = frozenset({"race", "sprint"})
_DISTANCE_REGIMES = frozenset({"full", "explicit_shortened", "no_points"})
_ELIGIBILITY_POLICIES = frozenset({"explicit_classification", "all_entered_engineering_assumption"})


def _season(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in _SUPPORTED_SEASONS:
        raise ValueError("supported championship seasons are 2019 through 2026")


def _number(value: float, name: str, *, nonnegative: bool = False) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return Fraction(str(value))


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _event_dict(event: EventId) -> dict[str, int]:
    return {"season": event.season, "round": event.round}


def _points_mapping(points: Mapping[str, float], kind: EntityKind) -> Mapping[str, float]:
    if not points:
        raise ValueError(f"{kind.value} standings must contain the complete championship roster")
    normalized: dict[str, float] = {}
    for identifier, value in points.items():
        EntityId(kind, identifier)
        _number(value, "current points")
        normalized[identifier] = float(value)
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class PointsRules:
    """Caller-selected scoring table for one season, event type and distance case.

    No lap-distance thresholds are inferred. A shortened event requires its
    selected table and a rules ID naming the exact case, or an explicit zero
    table for ``no_points``. This is a scoring scenario, not a rules engine.
    """

    season: int
    event_kind: str
    points_by_position: tuple[float, ...]
    rules_id: str
    source_url: str
    distance_regime: str = "full"
    fastest_lap_points: float = 0
    fastest_lap_max_position: int | None = None

    def __post_init__(self) -> None:
        _season(self.season)
        if self.event_kind not in _EVENT_KINDS:
            raise ValueError("event_kind must be race or sprint")
        if self.event_kind == "sprint" and self.season < 2021:
            raise ValueError("sprint scoring is unsupported before 2021")
        if self.distance_regime not in _DISTANCE_REGIMES:
            raise ValueError("distance_regime must explicitly select full, shortened or no points")
        table = tuple(self.points_by_position)
        if not table:
            raise ValueError("an explicit nonempty points table is required")
        for points in table:
            _number(points, "position points", nonnegative=True)
        if any(first < second for first, second in zip(table, table[1:], strict=False)):
            raise ValueError("position points must be nonincreasing")
        bonus = _number(self.fastest_lap_points, "fastest lap points", nonnegative=True)
        if bonus and (self.season >= 2025 or self.event_kind != "race"):
            raise ValueError("fastest-lap bonuses are unsupported for sprint or 2025 onward")
        if bonus and (
            isinstance(self.fastest_lap_max_position, bool)
            or not isinstance(self.fastest_lap_max_position, int)
            or self.fastest_lap_max_position != 10
        ):
            raise ValueError("2019-2024 fastest-lap eligibility must explicitly be top ten")
        if bonus and bonus != 1:
            raise ValueError("2019-2024 supported fastest-lap bonus is one point")
        if not bonus and self.fastest_lap_max_position is not None:
            raise ValueError("fastest-lap eligibility is ambiguous without a bonus")
        if self.distance_regime == "no_points" and (any(table) or bonus):
            raise ValueError("no_points requires zero position points and no fastest-lap bonus")
        _text(self.rules_id, "rules_id")
        _text(self.source_url, "source_url")
        object.__setattr__(self, "points_by_position", table)

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "event_kind": self.event_kind,
            "points_by_position": list(self.points_by_position),
            "rules_id": self.rules_id,
            "source_url": self.source_url,
            "distance_regime": self.distance_regime,
            "fastest_lap_points": self.fastest_lap_points,
            "fastest_lap_max_position": self.fastest_lap_max_position,
        }


@dataclass(frozen=True, slots=True)
class CurrentStandings:
    """Complete known points, including zero-point entrants and prior sanctions."""

    season: int
    available_at: datetime
    driver_points: Mapping[str, float]
    constructor_points: Mapping[str, float]
    source_hash: str

    def __post_init__(self) -> None:
        _season(self.season)
        require_utc(self.available_at, "standings available_at")
        _sha256(self.source_hash, "standings source_hash")
        object.__setattr__(
            self, "driver_points", _points_mapping(self.driver_points, EntityKind.DRIVER)
        )
        object.__setattr__(
            self,
            "constructor_points",
            _points_mapping(self.constructor_points, EntityKind.CONSTRUCTOR),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "available_at": self.available_at.isoformat(),
            "driver_points": dict(self.driver_points),
            "constructor_points": dict(self.constructor_points),
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True, slots=True)
class EventSimulation:
    """Complete joint orders and corresponding eligibility from one event model.

    ``points_eligible_samples`` contains canonical driver IDs, not positional
    masks. It must be supplied explicitly even when all drivers are assumed
    eligible. A sampled retirement is not itself FIA points ineligibility.
    Constructor assignment is event-specific to support replacement drivers and
    driver transfers without retroactively reallocating existing team points.
    """

    event_id: EventId
    scheduled_at: datetime
    available_at: datetime
    driver_constructors: Mapping[str, str]
    sampled_orders: tuple[tuple[str, ...], ...]
    points_eligible_samples: tuple[tuple[str, ...], ...]
    rules: PointsRules
    source_hash: str
    model_id: str = "engineering_fixture"
    eligibility_policy: str = "explicit_classification"
    fastest_lap_drivers: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, EventId):
            raise ValueError("event_id must be a canonical EventId")
        require_utc(self.scheduled_at, "scheduled_at")
        require_utc(self.available_at, "event available_at")
        if self.rules.season != self.event_id.season:
            raise ValueError("event and scoring-rule seasons must match")
        _sha256(self.source_hash, "event source_hash")
        _text(self.model_id, "model_id")
        if self.eligibility_policy not in _ELIGIBILITY_POLICIES:
            raise ValueError("points eligibility policy must be explicit")
        constructors = dict(sorted(self.driver_constructors.items()))
        if not constructors:
            raise ValueError("event requires a complete entered roster")
        for driver, constructor in constructors.items():
            EntityId(EntityKind.DRIVER, driver)
            EntityId(EntityKind.CONSTRUCTOR, constructor)
        orders = tuple(tuple(order) for order in self.sampled_orders)
        eligible = tuple(tuple(sorted(sample)) for sample in self.points_eligible_samples)
        if not orders or len(orders) != len(eligible):
            raise ValueError("each joint order requires a corresponding points-eligibility sample")
        roster = set(constructors)
        for order, classification in zip(orders, eligible, strict=True):
            if len(order) != len(roster) or set(order) != roster:
                raise ValueError("every sampled order must be a complete permutation of the roster")
            if len(classification) != len(set(classification)) or not set(classification) <= roster:
                raise ValueError("points-eligible drivers must be a unique subset of the roster")
            if self.eligibility_policy == "all_entered_engineering_assumption":
                if set(classification) != roster:
                    raise ValueError(
                        "all-entered eligibility assumption must include the full roster"
                    )
        laps = self.fastest_lap_drivers
        if self.rules.fastest_lap_points:
            if laps is None or len(laps) != len(orders):
                raise ValueError(
                    "fastest-lap scoring requires one explicit driver or None per sample"
                )
            if any(driver is not None and driver not in roster for driver in laps):
                raise ValueError("fastest-lap drivers must belong to the event roster")
        elif laps is not None:
            raise ValueError("fastest-lap samples are ambiguous when the rule awards no bonus")
        object.__setattr__(self, "driver_constructors", MappingProxyType(constructors))
        object.__setattr__(self, "sampled_orders", orders)
        object.__setattr__(self, "points_eligible_samples", eligible)
        if laps is not None:
            object.__setattr__(self, "fastest_lap_drivers", tuple(laps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": _event_dict(self.event_id),
            "scheduled_at": self.scheduled_at.isoformat(),
            "available_at": self.available_at.isoformat(),
            "driver_constructors": dict(self.driver_constructors),
            "sampled_orders": [list(order) for order in self.sampled_orders],
            "points_eligible_samples": [list(sample) for sample in self.points_eligible_samples],
            "rules": self.rules.to_dict(),
            "source_hash": self.source_hash,
            "model_id": self.model_id,
            "eligibility_policy": self.eligibility_policy,
            "fastest_lap_drivers": (
                None if self.fastest_lap_drivers is None else list(self.fastest_lap_drivers)
            ),
        }


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """A provenance-bound, independently validated Gold model gate.

    The simulator checks this contract; it does not independently establish
    model accuracy or audit a supplied report's statistical findings.
    """

    model_id: str
    model_validated: bool
    evidence_tier: str
    independent_gold_events: tuple[EventId, ...]
    available_at: datetime
    source_hash: str

    def __post_init__(self) -> None:
        _text(self.model_id, "validation model_id")
        if not isinstance(self.model_validated, bool):
            raise ValueError("model_validated must be Boolean")
        if self.evidence_tier not in {"Gold", "Silver", "Development"}:
            raise ValueError("validation tier must be Gold, Silver or Development")
        require_utc(self.available_at, "validation available_at")
        _sha256(self.source_hash, "validation source_hash")
        events = tuple(self.independent_gold_events)
        if any(not isinstance(event, EventId) for event in events):
            raise ValueError("validation events must be canonical EventIds")
        if len(events) != len(set(events)):
            raise ValueError("validation races must be distinct")
        object.__setattr__(self, "independent_gold_events", tuple(sorted(events)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_validated": self.model_validated,
            "evidence_tier": self.evidence_tier,
            "independent_gold_events": [
                _event_dict(event) for event in self.independent_gold_events
            ],
            "available_at": self.available_at.isoformat(),
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True, slots=True)
class TitleProbabilities:
    """Sole-title probability and unresolved tie mass, without invented countback.

    ``final_position_probability`` lists each entity's probability of finishing
    first, second and so on. Entities tied on points share the tied positions
    equally, so every row and every position column sums to one.
    """

    title_probability: Mapping[str, float]
    tied_for_title_probability: Mapping[str, float]
    unresolved_tie_probability: float
    mean_final_points: Mapping[str, float]
    final_position_probability: Mapping[str, tuple[float, ...]] = MappingProxyType({})

    def to_dict(self) -> dict[str, Any]:
        return {
            "title_probability": dict(self.title_probability),
            "tied_for_title_probability": dict(self.tied_for_title_probability),
            "unresolved_tie_probability": self.unresolved_tie_probability,
            "mean_final_points": dict(self.mean_final_points),
            "final_position_probability": {
                identifier: list(values)
                for identifier, values in self.final_position_probability.items()
            },
        }


@dataclass(frozen=True, slots=True)
class ChampionshipResult:
    season: int
    prediction_timestamp: datetime
    simulations: int
    seed: int
    status: str
    wdc: TitleProbabilities
    wcc: TitleProbabilities
    provenance: Mapping[str, Any]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "season": self.season,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "simulations": self.simulations,
            "seed": self.seed,
            "status": self.status,
            "wdc": self.wdc.to_dict(),
            "wcc": self.wcc.to_dict(),
            "provenance": dict(self.provenance),
            "limitations": list(self.limitations),
            "simulation_standard_error_max": 0.5 / math.sqrt(self.simulations),
        }
        return {**payload, "sha256": _digest(payload)}

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _summary(
    identifiers: tuple[str, ...],
    winners: list[int],
    tied: list[int],
    unresolved: int,
    totals: list[int],
    places: list[list[Fraction]],
    simulations: int,
    scale: int,
) -> TitleProbabilities:
    return TitleProbabilities(
        title_probability=MappingProxyType(
            {
                identifier: count / simulations
                for identifier, count in zip(identifiers, winners, strict=True)
            }
        ),
        tied_for_title_probability=MappingProxyType(
            {
                identifier: count / simulations
                for identifier, count in zip(identifiers, tied, strict=True)
            }
        ),
        unresolved_tie_probability=unresolved / simulations,
        mean_final_points=MappingProxyType(
            {
                identifier: float(Fraction(total, simulations * scale))
                for identifier, total in zip(identifiers, totals, strict=True)
            }
        ),
        final_position_probability=MappingProxyType(
            {
                identifier: tuple(float(value / simulations) for value in row)
                for identifier, row in zip(identifiers, places, strict=True)
            }
        ),
    )


def _record_places(final: list[int], places: list[list[Fraction]]) -> None:
    """Split tied championship positions evenly instead of inventing countback."""
    ordered = sorted(range(len(final)), key=lambda index: -final[index])
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and final[ordered[end + 1]] == final[ordered[start]]:
            end += 1
        share = Fraction(1, end - start + 1)
        for index in ordered[start : end + 1]:
            for position in range(start, end + 1):
                places[index][position] += share
        start = end + 1


def simulate_championship(
    standings: CurrentStandings,
    events: Sequence[EventSimulation],
    *,
    prediction_timestamp: datetime,
    simulations: int = 10000,
    seed: int = 42,
    validation: ValidationEvidence | None = None,
) -> ChampionshipResult:
    """Resample whole joint orders uniformly, independently between future events.

    Repeated orders represent their empirical probability mass. Sample order,
    classification eligibility and fastest-lap outcome always travel together.
    Scoring uses exact rational units so fractional-point ties are exact. This
    function neither samples marginals nor infers FIA eligibility from DNF.
    """
    require_utc(prediction_timestamp, "prediction_timestamp")
    require_known_by(standings.available_at, prediction_timestamp)
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or not 1 <= simulations <= 1000000
    ):
        raise ValueError("simulations must be an integer from 1 to 1000000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer from zero through 2**64 - 1")
    ordered_events = tuple(
        sorted(
            events, key=lambda event: (event.scheduled_at, event.event_id, event.rules.event_kind)
        )
    )
    identities: set[tuple[EventId, str]] = set()
    for event in ordered_events:
        require_known_by(event.available_at, prediction_timestamp)
        if event.scheduled_at <= prediction_timestamp:
            raise ValueError("remaining events must occur after prediction_timestamp")
        if event.event_id.season != standings.season:
            raise ValueError("every remaining event must belong to the standings season")
        identity = (event.event_id, event.rules.event_kind)
        if identity in identities:
            raise ValueError("duplicate remaining race or sprint")
        identities.add(identity)
        if not set(event.driver_constructors) <= set(standings.driver_points):
            raise ValueError("standings must include all remaining drivers, including zero points")
        if not set(event.driver_constructors.values()) <= set(standings.constructor_points):
            raise ValueError("standings must explicitly include all remaining constructors")
    status = "engineering_only"
    if validation is not None:
        require_known_by(validation.available_at, prediction_timestamp)
        if validation.model_validated:
            if (
                validation.evidence_tier != "Gold"
                or len(validation.independent_gold_events) < SELECTION_PAIRED_EVENTS
            ):
                raise ValueError(
                    "validated championship scenarios require 25 independent Gold races"
                )
            if any(event.model_id != validation.model_id for event in ordered_events):
                raise ValueError("Gold validation must bind every remaining event's model_id")
            if any(
                event.event_id in validation.independent_gold_events for event in ordered_events
            ):
                raise ValueError(
                    "validation races must be independent of remaining prediction events"
                )
            earliest_round = min((event.event_id.round for event in ordered_events), default=None)
            if any(
                event.season > standings.season
                or event.season > prediction_timestamp.year
                or (
                    event.season == standings.season
                    and earliest_round is not None
                    and event.round >= earliest_round
                )
                for event in validation.independent_gold_events
            ):
                raise ValueError("Gold validation cannot include a future season or later round")
            if any(
                event.eligibility_policy != "explicit_classification" for event in ordered_events
            ):
                raise ValueError(
                    "validated scenarios require explicit modeled points classification"
                )
            status = "validated_model_scenario"
    driver_ids = tuple(standings.driver_points)
    constructor_ids = tuple(standings.constructor_points)
    driver_indices = {identifier: index for index, identifier in enumerate(driver_ids)}
    constructor_indices = {identifier: index for index, identifier in enumerate(constructor_ids)}
    fractions = [Fraction(str(value)) for value in standings.driver_points.values()]
    fractions.extend(Fraction(str(value)) for value in standings.constructor_points.values())
    for event in ordered_events:
        fractions.extend(Fraction(str(value)) for value in event.rules.points_by_position)
        fractions.append(Fraction(str(event.rules.fastest_lap_points)))
    scale = math.lcm(*(value.denominator for value in fractions))

    def units(value: float) -> int:
        return int(Fraction(str(value)) * scale)

    current_drivers = [units(standings.driver_points[driver]) for driver in driver_ids]
    current_constructors = [units(standings.constructor_points[team]) for team in constructor_ids]
    event_scores: list[list[tuple[tuple[int, int, int], ...]]] = []
    for event in ordered_events:
        samples: list[tuple[tuple[int, int, int], ...]] = []
        for sample_index, (order, eligible) in enumerate(
            zip(event.sampled_orders, event.points_eligible_samples, strict=True)
        ):
            awards: list[tuple[int, int, int]] = []
            for position, driver in enumerate(order):
                points = 0
                if driver in eligible:
                    if position < len(event.rules.points_by_position):
                        points = units(event.rules.points_by_position[position])
                    if (
                        event.fastest_lap_drivers is not None
                        and driver == event.fastest_lap_drivers[sample_index]
                        and event.rules.fastest_lap_max_position is not None
                        and position < event.rules.fastest_lap_max_position
                    ):
                        points += units(event.rules.fastest_lap_points)
                awards.append(
                    (
                        driver_indices[driver],
                        constructor_indices[event.driver_constructors[driver]],
                        points,
                    )
                )
            samples.append(tuple(awards))
        event_scores.append(samples)
    rng = random.Random(seed)
    driver_wins, driver_ties, driver_totals = ([0] * len(driver_ids) for _ in range(3))
    team_wins, team_ties, team_totals = ([0] * len(constructor_ids) for _ in range(3))
    driver_places = [[Fraction(0)] * len(driver_ids) for _ in driver_ids]
    team_places = [[Fraction(0)] * len(constructor_ids) for _ in constructor_ids]
    driver_unresolved = team_unresolved = 0
    for _ in range(simulations):
        final_drivers = current_drivers.copy()
        final_teams = current_constructors.copy()
        for samples in event_scores:
            for driver_index, team_index, points in samples[rng.randrange(len(samples))]:
                final_drivers[driver_index] += points
                final_teams[team_index] += points
        _record_places(final_drivers, driver_places)
        _record_places(final_teams, team_places)
        for final, wins, ties, totals, kind in (
            (final_drivers, driver_wins, driver_ties, driver_totals, "driver"),
            (final_teams, team_wins, team_ties, team_totals, "constructor"),
        ):
            maximum = max(final)
            leaders = [index for index, points in enumerate(final) if points == maximum]
            if len(leaders) == 1:
                wins[leaders[0]] += 1
            else:
                for index in leaders:
                    ties[index] += 1
                if kind == "driver":
                    driver_unresolved += 1
                else:
                    team_unresolved += 1
            for index, points in enumerate(final):
                totals[index] += points
    event_payloads = [event.to_dict() for event in ordered_events]
    validation_payload = None if validation is None else validation.to_dict()
    input_payload = {
        "standings": standings.to_dict(),
        "events": event_payloads,
        "prediction_timestamp": prediction_timestamp.isoformat(),
        "simulations": simulations,
        "seed": seed,
        "validation": validation_payload,
    }
    provenance = {
        "input_sha256": _digest(input_payload),
        "standings_sha256": _digest(standings.to_dict()),
        "standings_source_hash": standings.source_hash,
        "event_sha256": [_digest(payload) for payload in event_payloads],
        "event_source_hashes": [event.source_hash for event in ordered_events],
        "rules_sha256": [_digest(event.rules.to_dict()) for event in ordered_events],
        "validation": validation_payload,
        "simulation_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampling": "uniform empirical joint-order resampling",
        "event_dependence": "independent between remaining race and sprint events",
        "ties": "unresolved; historical FIA countback is not available",
        "points_arithmetic": "exact rational units",
        "remaining_event_count": len(ordered_events),
        "eligibility_policies": [event.eligibility_policy for event in ordered_events],
    }
    limitations = (
        "Engineering fixtures do not establish predictive accuracy.",
        "Complete calendar, rosters, current points and scoring cases are caller-supplied.",
        "Points classification, fastest laps and shortened distances are explicit inputs.",
        "Remaining events are independent; shared season-level form and incident risk are omitted.",
        "Top-points ties are unresolved because historical FIA countback is not supplied.",
        "Monte Carlo standard error describes sampling error, not model uncertainty.",
    )
    return ChampionshipResult(
        season=standings.season,
        prediction_timestamp=prediction_timestamp,
        simulations=simulations,
        seed=seed,
        status=status,
        wdc=_summary(
            driver_ids,
            driver_wins,
            driver_ties,
            driver_unresolved,
            driver_totals,
            driver_places,
            simulations,
            scale,
        ),
        wcc=_summary(
            constructor_ids,
            team_wins,
            team_ties,
            team_unresolved,
            team_totals,
            team_places,
            simulations,
            scale,
        ),
        provenance=MappingProxyType(provenance),
        limitations=limitations,
    )
