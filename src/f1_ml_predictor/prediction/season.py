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
    ledger: ScoringLedger,
    target: EventId,
    cutoff: datetime,
    roster: dict[str, str],
    *,
    current_sprint: dict[str, Any] | None = None,
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
    if current_sprint is not None:
        # The current round's sprint has run but its FIA points belong to an event
        # version that is only complete after the race. Use the captured sprint
        # classification, credited to each driver's current constructor.
        captured = datetime.fromisoformat(current_sprint["captured_at"])
        if captured > cutoff:
            raise ValueError("sprint result capture is later than the cutoff")
        rule = ledger.rule_for(target)
        team_scoring = rule is None or rule.constructor_scoring == "sum_awarded_entries"
        season_rule, _ = _season_rule(ledger, target)
        awarded = sorted((p for p in current_sprint["points"].values() if p), reverse=True)
        if awarded != [float(p) for p in season_rule.sprint_points[: len(awarded)]]:
            raise ValueError("captured sprint points do not match the audited sprint table")
        sprint_points: dict[str, float] = current_sprint["points"]
        evidence, status = current_sprint["bundle"]["sha256"], "captured_live_development"
        provider = "OpenF1" if current_sprint.get("provider") == "openf1" else "Jolpica"
        note = (
            f"{target.partition()} sprint points come from the captured {provider} sprint "
            "classification (Development) until the FIA after-sprint audit"
        )
        fia = current_sprint.get("fia")
        if fia is not None:
            if datetime.fromisoformat(fia["captured_at"]) > cutoff:
                raise ValueError("FIA sprint capture is later than the cutoff")
            table = season_rule.sprint_points
            sprint_points = {
                driver: float(table[position - 1])
                if position is not None and position <= len(table)
                else 0.0
                for driver, position in fia["positions"].items()
            }
            differ = sorted(
                driver
                for driver in set(sprint_points) | set(current_sprint["points"])
                if sprint_points.get(driver, 0.0) != current_sprint["points"].get(driver, 0.0)
            )
            evidence, status = fia["bundle"]["sha256"], "captured_live_fia"
            note = (
                f"{target.partition()} sprint points come from the captured FIA Final Sprint "
                f"Classification (document {fia['document_id']}) and the audited sprint table"
                + (
                    f"; the {provider} capture differed for {', '.join(differ)}"
                    if differ
                    else f"; the {provider} capture agrees"
                )
            )
        for driver, points in sprint_points.items():
            drivers[driver] += points
            if team_scoring and driver in roster:
                constructors[roster[driver]] += points
        notes.append(note)
        used.append(
            {
                "event_id": f"{target.partition()}#sprint",
                "effective_at": captured.isoformat(),
                "revision_status": status,
                "evidence_hash": evidence,
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
    schedule: list[ScheduledEvent],
    target: ScheduledEvent,
    cutoff: datetime,
    *,
    sprint_counted: bool = False,
) -> list[tuple[ScheduledEvent, str, datetime]]:
    """List every race and sprint after the cutoff in this season, in time order.

    The target weekend's sprint may already have run when its points are counted
    in the starting standings (``sprint_counted``).
    """
    sessions = []
    for item in schedule:
        if item.event.season != target.event.season or item.race_start <= cutoff:
            continue
        if item.sprint_start is not None:
            if item.sprint_start <= cutoff:
                if not (sprint_counted and item.event == target.event):
                    raise ValueError(
                        f"{item.event.partition()} sprint has run but its points are not counted"
                    )
            else:
                sessions.append((item, "sprint", item.sprint_start))
        sessions.append((item, "race", item.race_start))
    if not sessions or sessions[0][0].event != target.event:
        raise ValueError("the next race must be the first remaining session")
    return sorted(sessions, key=lambda value: value[2])
