"""Development WDC and WCC scenarios built only from development race samples."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.live_features import ScheduledEvent
from f1_ml_predictor.scoring.ledger import ScoringLedger, ScoringRule
from f1_ml_predictor.simulation import CurrentStandings, PointsRules

ELIGIBILITY_POLICY = "all_entered_engineering_assumption"


def published_standings(
    ledger: ScoringLedger, target: EventId, cutoff: datetime, roster: dict[str, str]
) -> tuple[CurrentStandings, list[str], list[dict[str, Any]]]:
    """Sum the latest published FIA points of every earlier round before the cutoff.

    Unlike the strict Gold feature gate, values under appeal are used as published
    and named in the notes, because a season simulation needs a starting total.
    A round without any published complete version blocks the simulation.
    """
    drivers: dict[str, float] = defaultdict(float)
    constructors: dict[str, float] = defaultdict(float)
    notes: list[str] = []
    used: list[dict[str, Any]] = []
    for number in range(1, target.round):
        event = EventId(target.season, number)
        versions = [
            version
            for version in ledger.events
            if version.event == event
            and version.completed_at < cutoff
            and version.effective_at < cutoff
        ]
        if not versions:
            raise ValueError(f"round {number} has no published points version before the cutoff")
        latest = max(versions, key=lambda version: version.effective_at)
        if not latest.complete or any(entry.total_points is None for entry in latest.entries):
            raise ValueError(f"round {number} has incomplete published points")
        if latest.revision_status not in {"audited", "revised"}:
            notes.append(
                f"round {number} uses published points whose status is {latest.revision_status}"
            )
        rule = ledger.rule_for(event)
        team_scoring = rule is not None and rule.constructor_scoring == "sum_awarded_entries"
        for entry in latest.entries:
            points = float(entry.total_points or 0.0)
            if entry.driver_id is not None:
                drivers[entry.driver_id] += points
            if entry.constructor_id is not None and team_scoring:
                constructors[entry.constructor_id] += points
        used.append(
            {
                "event_id": event.partition(),
                "effective_at": latest.effective_at.isoformat(),
                "revision_status": latest.revision_status,
                "evidence_hash": latest.evidence_hash,
            }
        )
    for driver, constructor in roster.items():
        drivers.setdefault(driver, 0.0)
        constructors.setdefault(constructor, 0.0)
    available = max((datetime.fromisoformat(item["effective_at"]) for item in used), default=cutoff)
    standings = CurrentStandings(
        target.season, available, dict(drivers), dict(constructors), ledger.sha256
    )
    return standings, notes, used


def _season_rule(ledger: ScoringLedger, event: EventId) -> tuple[ScoringRule, bool]:
    rule = ledger.rule_for(event)
    if rule is not None:
        return rule, True
    rules = ledger.rules.get(event.season, ())
    earlier = [item for item in rules if item.last_round < event.round]
    if not earlier:
        raise ValueError(f"no audited scoring rule exists for season {event.season}")
    return max(earlier, key=lambda item: item.last_round), False


def points_rules(ledger: ScoringLedger, event: EventId, kind: str) -> tuple[PointsRules, str]:
    """Build explicit full-distance tables from the audited season rule.

    A round after the audited rule interval continues that season's latest rule.
    The returned note says so; shortened distances and cancellations are omitted.
    """
    rule, covered = _season_rule(ledger, event)
    if rule.fastest_lap_points:
        raise ValueError("fastest-lap seasons need explicit fastest-lap samples")
    table = rule.race_points if kind == "race" else rule.sprint_points
    if not table:
        raise ValueError(f"season {event.season} has no {kind} points table")
    status = (
        "audited_rule"
        if covered
        else f"continuation_of_audited_rounds_{rule.first_round}_{rule.last_round}"
    )
    source = rule.source_evidence[0]["reference"] if rule.source_evidence else "unknown"
    return (
        PointsRules(
            event.season,
            kind,
            tuple(table),
            f"ledger-rule-{rule.evidence_hash[:16]}-{kind}-full-{status}",
            source,
        ),
        status,
    )


def remaining_sessions(
    schedule: list[ScheduledEvent], target: ScheduledEvent, cutoff: datetime
) -> list[tuple[ScheduledEvent, str, datetime]]:
    """List every race and sprint after the cutoff in this season, in time order."""
    sessions = []
    for item in schedule:
        if item.event.season != target.event.season or item.race_start <= cutoff:
            continue
        if item.sprint_start is not None:
            if item.sprint_start <= cutoff:
                raise ValueError(
                    f"{item.event.partition()} sprint has run but its points are not audited"
                )
            sessions.append((item, "sprint", item.sprint_start))
        sessions.append((item, "race", item.race_start))
    if not sessions or sessions[0][0].event != target.event:
        raise ValueError("the next race must be the first remaining session")
    return sorted(sessions, key=lambda value: value[2])
