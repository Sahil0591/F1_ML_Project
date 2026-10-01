"""Validate audited point awards and reconstruct historical standings.

The ledger consumes awarded points, never finish positions. Each event version is a
complete replacement classification effective at its stated publication time.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_utc

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENTS = ("race_points", "sprint_points", "bonus_points", "adjustment_points")
_STATUSES = {"audited", "revised", "unknown", "disputed"}
_WINDOWS = (3, 5, 10)
SCORING_LEDGER_VERSION = "fia-timeline-v2"


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")
    return value


def _instant(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
        require_utc(parsed, label)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be an ISO UTC timestamp") from exc
    return parsed


def _points(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric points")
    if round(value * 4) != value * 4:
        raise ValueError(f"{label} must use quarter-point precision")
    return float(value)


def _evidence(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("source_evidence must contain at least one source")
    result = []
    for source in value:
        source = _object(source, {"reference", "sha256"}, "source evidence")
        if not isinstance(source["reference"], str) or not source["reference"].strip():
            raise ValueError("source reference is required")
        if not isinstance(source["sha256"], str) or not _SHA.fullmatch(source["sha256"]):
            raise ValueError("source sha256 is invalid")
        result.append(source)
    return tuple(result)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ScoringRule:
    season: int
    first_round: int
    last_round: int
    race_points: tuple[float, ...]
    sprint_points: tuple[float, ...]
    reduced_race_points: tuple[tuple[float, ...], ...]
    fastest_lap_points: float
    fastest_lap_eligible_through: int | None
    constructor_scoring: str
    source_evidence: tuple[dict[str, str], ...]
    evidence_hash: str
    revision_status: str


@dataclass(frozen=True, slots=True)
class EventPoints:
    season: int
    event_id: str
    driver_id: str | None
    constructor_id: str | None
    race_points: float | None
    sprint_points: float | None
    bonus_points: float | None
    adjustment_points: float | None
    total_points: float | None
    effective_at: datetime
    source_evidence: tuple[dict[str, str], ...]
    evidence_hash: str
    revision_status: str
    race_position: int | None = None
    race_position_audited: bool = False


@dataclass(frozen=True, slots=True)
class EventVersion:
    event: EventId
    completed_at: datetime
    effective_at: datetime
    race_schedule: str
    complete: bool
    revision_status: str
    entries: tuple[EventPoints, ...]
    evidence_hash: str


@dataclass(frozen=True, slots=True)
class ScoringLedger:
    rules: dict[int, tuple[ScoringRule, ...]]
    events: tuple[EventVersion, ...]
    sha256: str
    rules_sha256: str
    evidence_sha256: str

    def rule_for(self, event: EventId) -> ScoringRule | None:
        matches = [
            rule
            for rule in self.rules.get(event.season, ())
            if rule.first_round <= event.round <= rule.last_round
        ]
        return matches[0] if len(matches) == 1 else None

    def standings_before(
        self, event: EventId, cutoff: datetime, drivers: dict[str, str]
    ) -> dict[str, dict[str, float | str | None]]:
        """Return features for a roster, or a missing reason for each field.

        All earlier rounds must have a complete, audited version effective before
        the cutoff. A later correction replaces the earlier entire event version.
        """
        require_utc(cutoff, "prediction cutoff")
        rule = self.rule_for(event)
        reasons = "scoring_rule_missing_or_uncertain"
        selected: list[EventVersion] = []
        if rule is not None and rule.revision_status in {"audited", "revised"}:
            reasons = "audited_prior_event_missing_or_uncertain"
            for number in range(1, event.round):
                prior_rule = self.rule_for(EventId(event.season, number))
                if prior_rule is None or prior_rule.revision_status not in {"audited", "revised"}:
                    break
                versions = [
                    version
                    for version in self.events
                    if version.event == EventId(event.season, number)
                    and version.completed_at < cutoff
                    and version.effective_at < cutoff
                ]
                if not versions:
                    break
                latest = max(versions, key=lambda version: version.effective_at)
                if not latest.complete or latest.revision_status not in {"audited", "revised"}:
                    break
                if any(entry.total_points is None for entry in latest.entries):
                    break
                selected.append(latest)
            else:
                reasons = ""
        for driver, constructor in drivers.items():
            EntityId(EntityKind.DRIVER, driver)
            EntityId(EntityKind.CONSTRUCTOR, constructor)
        names = (
            "driver_points_before_race",
            "driver_championship_position",
            "driver_points_gap_to_leader",
            "constructor_points_before_race",
            "constructor_championship_position",
            "constructor_points_gap_to_leader",
            *(f"driver_points_last_{window}" for window in _WINDOWS),
            *(f"constructor_points_last_{window}" for window in _WINDOWS),
        )
        if reasons:
            return {
                driver: {**dict.fromkeys(names), "missing_reason": reasons} for driver in drivers
            }
        driver_totals: dict[str, float] = dict.fromkeys(drivers, 0.0)
        constructor_totals: dict[str, float] = dict.fromkeys(drivers.values(), 0.0)
        driver_places: dict[str, dict[int, int]] = defaultdict(dict)
        constructor_places: dict[str, dict[int, int]] = defaultdict(dict)
        countback_audited = True
        event_driver: list[dict[str, float]] = []
        event_constructor: list[dict[str, float]] = []
        for version in selected:
            d_points: dict[str, float] = {}
            c_points: dict[str, float] = {}
            event_rule = self.rule_for(version.event)
            for entry in version.entries:
                assert entry.total_points is not None
                countback_audited &= entry.race_position_audited
                if entry.driver_id is not None:
                    d_points[entry.driver_id] = (
                        d_points.get(entry.driver_id, 0.0) + entry.total_points
                    )
                    driver_totals[entry.driver_id] = (
                        driver_totals.get(entry.driver_id, 0.0) + entry.total_points
                    )
                    if entry.race_position is not None:
                        places = driver_places[entry.driver_id]
                        places[entry.race_position] = places.get(entry.race_position, 0) + 1
                if (
                    entry.constructor_id is not None
                    and event_rule is not None
                    and event_rule.constructor_scoring == "sum_awarded_entries"
                ):
                    c_points[entry.constructor_id] = (
                        c_points.get(entry.constructor_id, 0.0) + entry.total_points
                    )
                    constructor_totals[entry.constructor_id] = (
                        constructor_totals.get(entry.constructor_id, 0.0) + entry.total_points
                    )
                    if entry.race_position is not None:
                        places = constructor_places[entry.constructor_id]
                        places[entry.race_position] = places.get(entry.race_position, 0) + 1
            event_driver.append(d_points)
            event_constructor.append(c_points)

        max_place = max(
            (
                place
                for counts in (*driver_places.values(), *constructor_places.values())
                for place in counts
            ),
            default=0,
        )

        def rank(
            totals: dict[str, float], places: dict[str, dict[int, int]], identity: str
        ) -> float | None:
            value = totals.get(identity, 0.0)
            if not countback_audited:
                if sum(other == value for other in totals.values()) > 1:
                    return None
                return float(1 + sum(other > value for other in totals.values()))

            def key(entity: str) -> tuple[float | int, ...]:
                return (
                    totals.get(entity, 0.0),
                    *(
                        places.get(entity, {}).get(position, 0)
                        for position in range(1, max_place + 1)
                    ),
                )

            own = key(identity)
            if sum(key(entity) == own for entity in totals) > 1:
                return None  # A further FIA nomination or qualifying countback is needed.
            return float(1 + sum(key(entity) > own for entity in totals))

        result: dict[str, dict[str, float | str | None]] = {}
        constructor_contested = (
            rule is not None and rule.constructor_scoring == "sum_awarded_entries"
        )
        for driver, constructor in drivers.items():
            dp = driver_totals.get(driver, 0.0)
            cp = constructor_totals.get(constructor, 0.0)
            result[driver] = {
                "driver_points_before_race": dp,
                "driver_championship_position": rank(driver_totals, driver_places, driver),
                "driver_points_gap_to_leader": max(driver_totals.values(), default=0.0) - dp,
                "constructor_points_before_race": cp if constructor_contested else None,
                "constructor_championship_position": (
                    rank(constructor_totals, constructor_places, constructor)
                    if constructor_contested
                    else None
                ),
                "constructor_points_gap_to_leader": (
                    max(constructor_totals.values(), default=0.0) - cp
                )
                if constructor_contested
                else None,
                **{
                    f"driver_points_last_{window}": sum(
                        points.get(driver, 0.0) for points in event_driver[-window:]
                    )
                    if len(event_driver) >= window
                    else None
                    for window in _WINDOWS
                },
                **{
                    f"constructor_points_last_{window}": sum(
                        points.get(constructor, 0.0) for points in event_constructor[-window:]
                    )
                    if constructor_contested and len(event_constructor) >= window
                    else None
                    for window in _WINDOWS
                },
                "missing_reason": None,
                "constructor_missing_reason": (
                    None if constructor_contested else "constructor_championship_not_contested"
                ),
            }
        return result


def load_scoring_ledger(rules_path: Path, evidence_path: Path) -> ScoringLedger:
    """Import the versioned audit files with exact schemas and source hashes."""
    rules_bytes, evidence_bytes = rules_path.read_bytes(), evidence_path.read_bytes()
    rules_doc = json.loads(rules_bytes)
    events_doc = json.loads(evidence_bytes)
    if (
        isinstance(rules_doc, dict)
        and isinstance(events_doc, dict)
        and rules_doc.get("schema_version") == "scoring-evidence-v1"
        and events_doc.get("schema_version") == "scoring-evidence-v1"
    ):
        from f1_ml_predictor.scoring.audit_adapter import load_native_audit

        return load_native_audit(rules_doc, events_doc, rules_bytes, evidence_bytes, evidence_path)
    rules_doc = _object(rules_doc, {"schema_version", "rules"}, "rules file")
    events_doc = _object(events_doc, {"schema_version", "events"}, "events file")
    if rules_doc["schema_version"] != 1 or events_doc["schema_version"] != 1:
        raise ValueError("unsupported scoring evidence schema version")
    if not isinstance(rules_doc["rules"], list) or not isinstance(events_doc["events"], list):
        raise ValueError("scoring rules and events must be arrays")
    rules: dict[int, list[ScoringRule]] = {}
    for raw in rules_doc["rules"]:
        raw = _object(
            raw,
            {
                "season",
                "first_round",
                "last_round",
                "race_points",
                "sprint_points",
                "reduced_race_points",
                "fastest_lap_points",
                "fastest_lap_eligible_through",
                "constructor_scoring",
                "source_evidence",
                "evidence_hash",
                "revision_status",
            },
            "scoring rule",
        )
        season = EventId(raw["season"], 1).season
        first_round = EventId(season, raw["first_round"]).round
        last_round = EventId(season, raw["last_round"]).round
        if last_round < first_round or any(
            first_round <= rule.last_round and rule.first_round <= last_round
            for rule in rules.get(season, [])
        ):
            raise ValueError("overlapping or reversed scoring rule round range")
        if not isinstance(raw["reduced_race_points"], list):
            raise ValueError("reduced_race_points must be an array")
        schedules = [raw["race_points"], raw["sprint_points"], *raw["reduced_race_points"]]
        if any(not isinstance(item, list) for item in schedules):
            raise ValueError("point schedules must be arrays")
        parsed = [
            tuple(cast(float, _points(value, "schedule points")) for value in item)
            for item in schedules
        ]
        if any(any(value is None or value < 0 for value in schedule) for schedule in parsed):
            raise ValueError("point schedules must be nonnegative")
        if raw["constructor_scoring"] not in {"sum_awarded_entries", "not_contested"}:
            raise ValueError("unsupported constructor scoring policy")
        lap = _points(raw["fastest_lap_points"], "fastest lap points")
        if lap is None or lap < 0:
            raise ValueError("fastest lap points must be nonnegative")
        limit = raw["fastest_lap_eligible_through"]
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("fastest lap eligibility must be a positive position or null")
        status = raw["revision_status"]
        if status not in _STATUSES:
            raise ValueError("invalid scoring rule revision status")
        sources = _evidence(raw["source_evidence"])
        if raw["evidence_hash"] != _digest(
            {key: value for key, value in raw.items() if key != "evidence_hash"}
        ):
            raise ValueError("scoring rule evidence hash mismatch")
        rules.setdefault(season, []).append(
            ScoringRule(
                season,
                first_round,
                last_round,
                parsed[0],
                parsed[1],
                tuple(parsed[2:]),
                lap,
                limit,
                raw["constructor_scoring"],
                sources,
                raw["evidence_hash"],
                status,
            )
        )
    events = []
    seen = set()
    for raw in events_doc["events"]:
        raw = _object(
            raw,
            {
                "season",
                "event_id",
                "completed_at",
                "effective_at",
                "race_schedule",
                "complete",
                "revision_status",
                "expected_driver_ids",
                "entries",
                "evidence_hash",
            },
            "event version",
        )
        event = EventId(raw["season"], int(str(raw["event_id"]).split("/round=")[-1]))
        if raw["event_id"] != event.partition() or event.season not in rules:
            raise ValueError("event identity or scoring rule is invalid")
        completed = _instant(raw["completed_at"], "event completion")
        effective = _instant(raw["effective_at"], "event effective time")
        if effective < completed or type(raw["complete"]) is not bool:
            raise ValueError("event completeness or chronology invalid")
        if raw["revision_status"] not in _STATUSES:
            raise ValueError("invalid event revision status")
        matching_rules = [
            rule
            for rule in rules[event.season]
            if rule.first_round <= event.round <= rule.last_round
        ]
        if len(matching_rules) != 1:
            raise ValueError("event lacks a unique season scoring rule")
        ruleset = matching_rules[0]
        schedule = raw["race_schedule"]
        if schedule == "standard":
            allowed_race = {0.0, *ruleset.race_points}
        elif isinstance(schedule, str) and re.fullmatch(r"reduced:[0-9]+", schedule):
            index = int(schedule.split(":")[1])
            if index >= len(ruleset.reduced_race_points):
                raise ValueError("event reduced scoring schedule is not in the season rule")
            allowed_race = {0.0, *ruleset.reduced_race_points[index]}
        else:
            raise ValueError("event race_schedule must select an audited scoring schedule")
        key = (event, effective)
        if key in seen:
            raise ValueError("duplicate event revision effective time")
        seen.add(key)
        if not isinstance(raw["entries"], list):
            raise ValueError("event entries must be an array")
        expected = raw["expected_driver_ids"]
        if (
            not isinstance(expected, list)
            or any(not isinstance(driver_id, str) for driver_id in expected)
            or len(expected) != len(set(expected))
        ):
            raise ValueError("event expected_driver_ids must be unique")
        for driver_id in expected:
            EntityId(EntityKind.DRIVER, driver_id)
        entries = []
        identities = set()
        for item in raw["entries"]:
            item = _object(
                item,
                {
                    "season",
                    "event_id",
                    "driver_id",
                    "constructor_id",
                    *_COMPONENTS,
                    "total_points",
                    "effective_at",
                    "source_evidence",
                    "evidence_hash",
                    "revision_status",
                },
                "event points",
            )
            driver, constructor = item["driver_id"], item["constructor_id"]
            if driver is None and constructor is None:
                raise ValueError("point award requires a driver or constructor")
            if driver is not None:
                EntityId(EntityKind.DRIVER, driver)
            if constructor is not None:
                EntityId(EntityKind.CONSTRUCTOR, constructor)
            if (driver, constructor) in identities:
                raise ValueError("duplicate point recipient in event revision")
            identities.add((driver, constructor))
            if item["season"] != event.season or item["event_id"] != event.partition():
                raise ValueError("point entry belongs to another event")
            if _instant(item["effective_at"], "point effective time") != effective:
                raise ValueError("point entry effective time differs from event revision")
            if item["revision_status"] != raw["revision_status"]:
                raise ValueError("point entry revision status differs from event revision")
            sources = _evidence(item["source_evidence"])
            values = [_points(item[name], name, nullable=True) for name in _COMPONENTS]
            total = _points(item["total_points"], "total_points", nullable=True)
            if all(value is not None for value in values):
                if total != sum(value for value in values if value is not None):
                    raise ValueError("awarded point components do not sum to total")
            elif total is not None:
                raise ValueError("known total cannot contain unknown components")
            if item["evidence_hash"] != _digest(
                {key: value for key, value in item.items() if key != "evidence_hash"}
            ):
                raise ValueError("point entry evidence hash mismatch")
            race, sprint, bonus, adjustment = values
            if race is not None and race not in allowed_race:
                raise ValueError("race award is outside audited season schedules")
            if sprint is not None and sprint not in {*ruleset.sprint_points, 0.0}:
                raise ValueError("sprint award is outside audited season schedule")
            if bonus is not None and bonus not in {0.0, ruleset.fastest_lap_points}:
                raise ValueError("bonus award is outside audited fastest-lap rule")
            if driver is None and any(value not in {0.0, None} for value in (race, sprint, bonus)):
                raise ValueError("constructor-only entry may contain adjustments only")
            entries.append(
                EventPoints(
                    event.season,
                    event.partition(),
                    driver,
                    constructor,
                    race,
                    sprint,
                    bonus,
                    adjustment,
                    total,
                    effective,
                    sources,
                    item["evidence_hash"],
                    item["revision_status"],
                )
            )
        if raw["complete"] and not entries:
            raise ValueError("complete event requires audited point entries")
        primary = [
            entry.driver_id
            for entry in entries
            if entry.driver_id is not None and entry.constructor_id is not None
        ]
        if raw["complete"] and (set(primary) != set(expected) or len(primary) != len(expected)):
            raise ValueError("complete event does not cover its audited driver roster")
        if raw["evidence_hash"] != _digest(
            {key: value for key, value in raw.items() if key != "evidence_hash"}
        ):
            raise ValueError("event version evidence hash mismatch")
        events.append(
            EventVersion(
                event,
                completed,
                effective,
                schedule,
                raw["complete"],
                raw["revision_status"],
                tuple(entries),
                raw["evidence_hash"],
            )
        )
    for event in {version.event for version in events}:
        completed_times = {version.completed_at for version in events if version.event == event}
        if len(completed_times) != 1:
            raise ValueError("event revisions disagree on original completion time")
    rules_hash = hashlib.sha256(rules_bytes).hexdigest()
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
    return ScoringLedger(
        {
            season: tuple(sorted(values, key=lambda rule: rule.first_round))
            for season, values in rules.items()
        },
        tuple(sorted(events, key=lambda item: (item.event, item.effective_at))),
        _digest(
            {
                "engine_version": SCORING_LEDGER_VERSION,
                "rules_sha256": rules_hash,
                "evidence_sha256": evidence_hash,
            }
        ),
        rules_hash,
        evidence_hash,
    )
