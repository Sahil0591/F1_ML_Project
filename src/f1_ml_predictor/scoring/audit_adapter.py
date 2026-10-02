"""Import Claude's retained FIA scoring audit without changing its source files."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.scoring.ledger import (
    SCORING_LEDGER_VERSION,
    EventPoints,
    EventVersion,
    ScoringLedger,
    ScoringRule,
    _digest,
    _instant,
    _points,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SEASON_ROUND_LIMIT = 99
_REDUCED = (
    "race_reduced_col1_2laps_to_lt25pct",
    "race_reduced_col2_25_to_lt50pct",
    "race_reduced_col3_50_to_lt75pct",
)


def _verified_file(root: Path, relative: str, digest: str) -> None:
    if not isinstance(digest, str) or not _SHA.fullmatch(digest):
        raise ValueError("audit source hash is invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("audited source file is missing or outside the workspace")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("audited source file hash mismatch")


def _schedule(rows: list[dict[str, Any]], season: int, kind: str) -> tuple[float, ...]:
    selected = sorted(
        (row for row in rows if row.get("season") == season and row.get("event_type") == kind),
        key=lambda row: row["position"],
    )
    if [row["position"] for row in selected] != list(range(1, len(selected) + 1)):
        raise ValueError("audited scoring positions are incomplete or duplicated")
    values = tuple(cast(float, _points(row["points"], "audited rule points")) for row in selected)
    if any(value < 0 for value in values):
        raise ValueError("audited scoring schedule has negative points")
    return values


def _place(token: Any) -> int | None:
    if token is None or token == "NC" or token == "DQ":
        return None
    if not isinstance(token, str) or re.fullmatch(r"[0-9]+F?", token) is None:
        raise ValueError("unsupported FIA race placing token")
    return int(token.removesuffix("F"))


def _rules(
    rules_doc: dict[str, Any], root: Path, summaries: list[dict[str, Any]]
) -> dict[int, tuple[ScoringRule, ...]]:
    raw_rules = rules_doc.get("rules")
    season_rules = rules_doc.get("season_rules")
    if not isinstance(raw_rules, list) or not isinstance(season_rules, list):
        raise ValueError("native scoring rules are incomplete")
    by_season: dict[int, tuple[ScoringRule, ...]] = {}
    for regime in season_rules:
        season = EventId(regime["season"], 1).season
        if season in by_season:
            raise ValueError("native audit has duplicate season rules")
        intervals = [row for row in summaries if row["season"] == season]
        if not intervals:
            raise ValueError("native season has no audited events")
        if "both cars" not in regime["constructor_scoring"]:
            raise ValueError("unsupported audited constructor scoring rule")
        race = _schedule(raw_rules, season, "race")
        sprint = _schedule(raw_rules, season, "sprint")
        reduced = tuple(_schedule(raw_rules, season, kind) for kind in _REDUCED)
        if not race or not sprint:
            raise ValueError("season race or sprint scoring schedule is absent")
        bonus_rows = [
            row
            for row in raw_rules
            if row.get("season") == season and row.get("event_type") == "fastest_lap_bonus"
        ]
        if len(bonus_rows) > 1:
            raise ValueError("duplicate fastest-lap bonus rule")
        bonus = (
            cast(float, _points(bonus_rows[0]["points"], "fastest-lap rule")) if bonus_rows else 0.0
        )
        if bool(bonus) != ("No fastest-lap point" not in regime["fastest_lap_rule"]):
            raise ValueError("fastest-lap rule text and point schedule disagree")
        sources = []
        for key in ("first_issue_in_force", "last_issue_checked"):
            source = regime["evidence_source"][key]
            _verified_file(
                root, f"data/raw/fia_audit/objects/{source['sha256']}.pdf", source["sha256"]
            )
            sources.append({"reference": source["url"], "sha256": source["sha256"]})
        # The regulation is in force for the whole season, so the rule also covers
        # rounds that have no published points yet, including the next race.
        if not str(regime["effective_to"]).startswith(f"{season}-"):
            raise ValueError("season scoring regulation does not run to the season end")
        by_season[season] = (
            ScoringRule(
                season=season,
                first_round=1,
                last_round=_SEASON_ROUND_LIMIT,
                race_points=race,
                sprint_points=sprint,
                reduced_race_points=reduced,
                fastest_lap_points=bonus,
                fastest_lap_eligible_through=10 if bonus else None,
                constructor_scoring="sum_awarded_entries",
                source_evidence=tuple(sources),
                evidence_hash=_digest(regime),
                revision_status="audited",
            ),
        )
    return by_season


def load_native_audit(
    rules_doc: dict[str, Any],
    evidence_doc: dict[str, Any],
    rules_bytes: bytes,
    evidence_bytes: bytes,
    evidence_path: Path,
) -> ScoringLedger:
    """Validate retained bytes and reconstruct complete event versions at each FIA clock."""
    if rules_doc.get("audit_as_of_utc") != evidence_doc.get("audit_as_of_utc"):
        raise ValueError("scoring rule and award audit dates disagree")
    if evidence_doc.get("unresolved_events") != []:
        raise ValueError("unresolved scoring events cannot enter Gold")
    root = evidence_path.resolve().parents[2]
    collection = evidence_doc.get("collection_index")
    if not isinstance(collection, dict):
        raise ValueError("native audit collection index is missing")
    _verified_file(root, collection["path"], collection["sha256"])
    summaries = evidence_doc.get("event_summaries")
    records = evidence_doc.get("records")
    documents = evidence_doc.get("documents")
    constructor_awards = evidence_doc.get("constructor_event_points")
    if not all(
        isinstance(value, list) for value in (summaries, records, documents, constructor_awards)
    ):
        raise ValueError("native scoring audit tables are missing")
    summaries = cast(list[dict[str, Any]], summaries)
    records = cast(list[dict[str, Any]], records)
    documents = cast(list[dict[str, Any]], documents)
    constructor_awards = cast(list[dict[str, Any]], constructor_awards)
    rules = _rules(rules_doc, root, summaries)
    doc_index = {}
    for document in documents:
        key = document["document_key"]
        if key in doc_index:
            raise ValueError("duplicate FIA scoring document")
        digest = document["sha256"]
        if digest is not None:
            if key != digest:
                raise ValueError("FIA scoring document key and hash disagree")
            _verified_file(root, f"data/raw/fia_audit/objects/{digest}.pdf", digest)
        elif not document["recalled"]:
            raise ValueError("unrecalled FIA scoring document has no retained bytes")
        doc_index[key] = document
    summary_index = {}
    for summary in summaries:
        event = EventId(summary["season"], summary["round"])
        if summary["event_id"] != event.partition() or summary["status"] != "resolved":
            raise ValueError("scoring event summary is unresolved or noncanonical")
        if event in summary_index:
            raise ValueError("duplicate scoring event summary")
        summary_index[event] = summary
    grouped: dict[EventId, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        event = EventId(record["season"], record["round"])
        if record["event_id_if_known"] != event.partition() or event not in summary_index:
            raise ValueError("scoring record has no matching event summary")
        EntityId(EntityKind.DRIVER, record["driver"])
        EntityId(EntityKind.CONSTRUCTOR, record["constructor"])
        rule = rules[event.season][0]
        race = cast(float, _points(record["race_points"], "race award"))
        sprint = cast(float, _points(record["sprint_points"], "sprint award"))
        bonus = cast(float, _points(record["bonus_points"], "bonus award"))
        total = cast(float, _points(record["total_event_points"], "event award"))
        if (
            race not in {0.0, *rule.race_points}
            or sprint not in {0.0, *rule.sprint_points}
            or bonus not in {0.0, rule.fastest_lap_points}
            or total != race + sprint + bonus
            or record["checks"].get("race_points_match_full_scale") is not True
        ):
            raise ValueError("native awarded components disagree with audited scoring rules")
        quality = record["evidence_quality"]
        if quality == "conflict_secondary_source":
            if record["checks"].get("jolpica_race_agrees") is not True:
                raise ValueError("unresolved secondary scoring conflict")
        elif quality not in {
            "fia_official_formula1_confirmed",
            "fia_later_document_only_formula1_confirmed",
            "fia_total_formula1_sprint_split",
            "fia_official_unconfirmed",
        }:
            raise ValueError("unsupported scoring evidence quality")
        timeline = record["points_timeline"]
        if not isinstance(timeline, list) or not timeline:
            raise ValueError("scoring record has no publication timeline")
        clocks = []
        for item in timeline:
            timeline_document = doc_index.get(item["document"])
            if (
                timeline_document is None
                or timeline_document["sha256"] is None
                or timeline_document["recalled"]
            ):
                raise ValueError("point timeline refers to an unverified FIA document")
            if (
                item["published_at_utc_upper_bound"]
                != timeline_document["published_at_utc_upper_bound"]
            ):
                raise ValueError("point publication clock differs from the FIA document")
            clocks.append(_instant(item["published_at_utc_upper_bound"], "point publication"))
            _points(item["points"], "timeline award")
        if clocks != sorted(set(clocks)):
            raise ValueError("point revisions must have distinct increasing publication clocks")
        if (
            record["first_published_at"] != timeline[0]["published_at_utc_upper_bound"]
            or record["published_at"] != timeline[-1]["published_at_utc_upper_bound"]
            or total != timeline[-1]["points"]
        ):
            raise ValueError("point timeline disagrees with final award metadata")
        grouped[event].append(record)
    if set(grouped) != set(summary_index):
        raise ValueError("some audited scoring events have no driver records")
    for event, rows in grouped.items():
        if len(rows) != summary_index[event]["drivers"] or len({r["driver"] for r in rows}) != len(
            rows
        ):
            raise ValueError("scoring event has an incomplete or duplicate driver roster")
    entrant_index = {}
    for item in constructor_awards:
        if item["agrees"] is not True:
            raise ValueError("FIA constructor matrix disagrees with driver awards")
        key = (item["event_id_if_known"], item["constructor"])
        if key in entrant_index:
            raise ValueError("duplicate FIA constructor award")
        entrant_index[key] = item
    calculated: dict[tuple[str, str], float] = defaultdict(float)
    for record in records:
        calculated[(record["event_id_if_known"], record["constructor"])] += record[
            "total_event_points"
        ]
    if set(calculated) != set(entrant_index) or any(
        value != entrant_index[key]["fia_entrant_event_points"] for key, value in calculated.items()
    ):
        raise ValueError("audited constructor awards do not reconcile with FIA matrices")
    versions = []
    for event, rows in sorted(grouped.items()):
        times = sorted(
            {
                _instant(item["published_at_utc_upper_bound"], "point publication")
                for row in rows
                for item in row["points_timeline"]
            }
        )
        for effective in times:
            entries = []
            revised = False
            for row in sorted(rows, key=lambda item: item["driver"]):
                available = [
                    (index, item)
                    for index, item in enumerate(row["points_timeline"])
                    if _instant(item["published_at_utc_upper_bound"], "point publication")
                    <= effective
                ]
                if not available:
                    continue
                index, item = available[-1]
                revised |= index > 0
                document = doc_index[item["document"]]
                same_as_final = index == len(row["points_timeline"]) - 1
                record_hash = _digest(row)
                sources = (
                    {
                        "reference": f"audit:{event.partition()}:{row['driver']}",
                        "sha256": record_hash,
                    },
                    {"reference": document["url"], "sha256": document["sha256"]},
                )
                entries.append(
                    EventPoints(
                        season=event.season,
                        event_id=event.partition(),
                        driver_id=row["driver"],
                        constructor_id=row["constructor"],
                        race_points=float(row["race_points"]) if same_as_final else None,
                        sprint_points=float(row["sprint_points"]) if same_as_final else None,
                        bonus_points=float(row["bonus_points"]) if same_as_final else None,
                        adjustment_points=0.0 if same_as_final else None,
                        total_points=float(item["points"]),
                        effective_at=effective,
                        source_evidence=sources,
                        evidence_hash=_digest({"record_sha256": record_hash, "timeline": item}),
                        revision_status="revised" if index else "audited",
                        race_position=_place(item["position_token"]),
                        race_position_audited=(
                            _place(row["race_position_fia"]) == _place(item["position_token"])
                        ),
                    )
                )
            complete = len(entries) == len(rows)
            disputed = any(row["revision_status"] == "pending_appeal" for row in rows)
            status = (
                "unknown"
                if not complete
                else "disputed"
                if disputed
                else "revised"
                if revised
                else "audited"
            )
            versions.append(
                EventVersion(
                    event=event,
                    completed_at=times[0],
                    effective_at=effective,
                    race_schedule="standard",
                    complete=complete,
                    revision_status=status,
                    entries=tuple(entries),
                    evidence_hash=_digest(
                        {
                            "event_id": event.partition(),
                            "effective_at": effective.isoformat(),
                            "status": status,
                            "entries": [entry.evidence_hash for entry in entries],
                        }
                    ),
                )
            )
    rules_hash = hashlib.sha256(rules_bytes).hexdigest()
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
    return ScoringLedger(
        rules=rules,
        events=tuple(versions),
        sha256=_digest(
            {
                "engine_version": SCORING_LEDGER_VERSION,
                "rules_sha256": rules_hash,
                "evidence_sha256": evidence_hash,
            }
        ),
        rules_sha256=rules_hash,
        evidence_sha256=evidence_hash,
    )
