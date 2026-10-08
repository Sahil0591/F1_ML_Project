"""Development sprint race predictions from sprint qualifying (addendum sprint-dev-v1).

The strength candidates are the Gold post-qualifying race models, trained on main
races whose labels are known by the sprint cutoff. A sprint row feeds the sprint
qualifying classification where the race contract reads qualifying. Sprints differ
from races in length and attrition, so calibration (temperature and prior mixing)
and the retirement rate are fitted on earlier sprints only, through the unchanged
protocol v3 ``analyse``.

The sprint history is Development tier: Jolpica sprint classifications and OpenF1
sprint qualifying results, retained as current-state downloads and cross-checked
against the FIA sprint points already audited in the scoring ledger. A sprint whose
sources disagree is excluded with a reason. Nothing here becomes Gold.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pyarrow.parquet as pq

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.candidates import (
    StrengthModel,
    baseline_marginals,
    fit_dnf,
    fit_strength,
)
from f1_ml_predictor.prediction.contracts import (
    PRACTICE_NUMERIC,
    SPRINT_NUMERIC,
    AuditedHistory,
    build_rows,
)
from f1_ml_predictor.prediction.evaluation import (
    RaceRecord,
    analyse,
    columns_for,
    training_rows,
)
from f1_ml_predictor.prediction.joint import marginals, sample_mixture
from f1_ml_predictor.prediction.protocol import (
    DNF_CANDIDATES,
    JOINT_CANDIDATES,
    MIN_TRAIN_EVENTS,
    PROTOCOL_V3_SHA256,
    PROTOCOL_V3_VERSION,
)
from f1_ml_predictor.sources.http import JsonSourceClient
from f1_ml_predictor.sources.jolpica import BASE_URL
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.sprint_gold import SPRINT_GOLD_VERSION, load_gold_sprints
from f1_ml_predictor.trust.winter import DRIVER_ALIASES

SPRINT_CONTRACT = "post_sprint_qualifying"
RACE_CONTRACT = "post_qualifying"
SPRINT_SESSION_NAMES = ("Sprint Qualifying", "Sprint Shootout")
OPENF1 = "https://api.openf1.org/v1"
# Practice is never captured live, so sprint rows never carry it; same-weekend sprint
# values describe the sprint being predicted, so they are hidden too.
MASKED = (*PRACTICE_NUMERIC, *SPRINT_NUMERIC)
# 2022 had no sprint qualifying session: Friday qualifying set the sprint grid.
# Its classification is bounded conservatively after the scheduled start.
LEGACY_GRID_BOUND = timedelta(hours=2)
# A sprint classification is bounded conservatively after the scheduled start.
SPRINT_LABEL_BOUND = timedelta(hours=2)
_RAW = Path("data/raw/sprint_sources")
_LIMITS = ((3, 1.0), (30, 60.0))
# Jolpica identifiers that differ from the canonical Gold identifiers.
JOLPICA_CONSTRUCTORS = {"alfa": "alfa_romeo", "alphatauri": "alpha_tauri"}
# OpenF1 team names in a live sprint qualifying capture, as canonical Gold constructors.
# A name missing here falls back to the latest audited roster.
OPENF1_TEAMS = {
    "Alpine": "alpine",
    "Aston Martin": "aston_martin",
    "Audi": "audi",
    "Cadillac": "cadillac",
    "Ferrari": "ferrari",
    "Haas F1 Team": "haas",
    "McLaren": "mclaren",
    "Mercedes": "mercedes",
    "Racing Bulls": "rb",
    "Red Bull Racing": "red_bull",
    "Williams": "williams",
}

SPRINT_ADDENDUM = {
    "version": "sprint-dev-v1",
    "base_protocol": PROTOCOL_V3_VERSION,
    "base_protocol_sha256": PROTOCOL_V3_SHA256,
    "tier": "Development",
    "target": "sprint race finishing order, winner, podium and retirement",
    "cutoff": "OpenF1 sprint qualifying (2024+) or sprint shootout (2023) session end; "
    "2022 scheduled qualifying start plus two hours, because Friday qualifying set the "
    "sprint grid",
    "strength_training": "protocol v3 joint candidates fitted on Gold post_qualifying race "
    "rows whose labels were published by the sprint cutoff",
    "sprint_inputs": {
        "qualifying_position": "sprint qualifying classification position",
        "qualifying_last_session_seconds": "lap time in the last sprint qualifying stage reached",
        "teammate_qualifying_position_delta": "own minus teammate sprint qualifying position",
    },
    "masked_predictors": list(MASKED),
    "calibration": "protocol v3 analyse over sprint out-of-fold records only: temperature, "
    "prior mixing and development primary are prequential on earlier sprints",
    "dnf": "prior is the smoothed retirement rate of earlier sprints; logistic and hist are "
    "fitted on race labels and compete on out-of-fold sprint Brier score",
    "labels": "Jolpica sprint classification; numeric position text is classified, "
    "retirements are DNF, DNS and DSQ are unlabelled; label time is the scheduled sprint "
    "start plus two hours",
    "cross_check": "every Jolpica sprint points value must equal the FIA sprint points in the "
    "audited scoring ledger, and every FIA sprint position must match",
    "baselines": ["heuristic", "logistic"],
    "selection": "protocol v3 formal gate; Development evidence never selects a validated model",
}
SPRINT_ADDENDUM_SHA256 = hashlib.sha256(
    json.dumps(SPRINT_ADDENDUM, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
# The same method on the FIA-audited Gold sprint history (trust/sprint_gold.py).
SPRINT_GOLD_ADDENDUM = {
    **SPRINT_ADDENDUM,
    "version": "sprint-gold-v2",
    "tier": "Gold",
    "cutoff": "FIA registry publication of the first non-recalled sprint grid document "
    "(sprint qualifying 2024+, sprint shootout 2023, qualifying 2022), read as the later "
    "UTC bound plus one minute, and published before the Final Sprint Starting Grid",
    "labels": "latest non-recalled FIA Final Sprint Classification with no later sprint "
    "ruling, except a ruling reviewed against that classification and found not to amend "
    "it, bound to the ruling PDF hash; label time is its publication upper bound plus one "
    "minute; retirements are labelled only on FIA, Jolpica and OpenF1 agreement under the "
    "binary DNF rule",
    "audit_method": "fia-sprint-direct-v2-cet-upper-bound",
    "cross_check": "every FIA sprint position must match the audited FIA sprint points ledger",
}
SPRINT_GOLD_ADDENDUM_SHA256 = hashlib.sha256(
    json.dumps(SPRINT_GOLD_ADDENDUM, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def addendum_for(tier: str) -> tuple[dict[str, Any], str]:
    if tier == "Gold":
        return SPRINT_GOLD_ADDENDUM, SPRINT_GOLD_ADDENDUM_SHA256
    return SPRINT_ADDENDUM, SPRINT_ADDENDUM_SHA256


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class SprintGrid:
    """Sprint qualifying values for one event and the time they were known."""

    positions: dict[str, int | None]
    last_seconds: dict[str, float | None]
    available_at: datetime
    source: str


@dataclass(frozen=True)
class SprintEvent:
    event: EventId
    circuit_id: str
    cutoff: datetime
    sprint_start: datetime
    roster: dict[str, str]
    grid: SprintGrid
    labels: dict[str, dict[str, Any]]


def _retained(
    root: Path,
    name: str,
    url: str,
    params: dict[str, Any],
    client: JsonSourceClient | None,
) -> Any:
    """Fetch a source once, then reuse the hash-verified retained payload."""
    path = root / _RAW / f"{name}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if hashlib.sha256(_canonical(record["payload"])).hexdigest() != record["payload_sha256"]:
            raise ValueError(f"retained sprint source {name} hash mismatch")
        return record["payload"]
    if client is None:
        raise ValueError(f"sprint source {name} is not retained")
    payload = client.get_json(url, params)
    record = {
        "version": 1,
        "url": url,
        "params": params,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "classification": "current_state_only",
        "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical(record))
    return payload


@dataclass(frozen=True)
class Crosswalk:
    """Jolpica identities for one season mapped to canonical Gold identifiers."""

    drivers: dict[str, str]
    codes: dict[str, str]

    def driver(self, jolpica_id: str) -> str:
        return self.drivers.get(jolpica_id, jolpica_id)

    @staticmethod
    def constructor(jolpica_id: str | None) -> str | None:
        return None if jolpica_id is None else JOLPICA_CONSTRUCTORS.get(jolpica_id, jolpica_id)


def crosswalk(payload: Any) -> Crosswalk:
    """Canonical IDs come from the FIA name table, as in Gold; codes match OpenF1 acronyms."""
    drivers: dict[str, str] = {}
    codes: dict[str, str] = {}
    for driver in payload["MRData"]["DriverTable"]["Drivers"]:
        name = f"{driver.get('givenName', '')} {str(driver.get('familyName', '')).upper()}"
        canonical = DRIVER_ALIASES.get(name, driver["driverId"])
        drivers[driver["driverId"]] = canonical
        code = driver.get("code")
        if not code:
            continue
        if codes.get(code, canonical) != canonical:
            raise ValueError(f"duplicate Jolpica driver code {code}")
        codes[code] = canonical
    return Crosswalk(drivers, codes)


def openf1_grid(
    results: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    codes: dict[str, str],
    available_at: datetime,
    source: str,
) -> SprintGrid:
    """Map an OpenF1 sprint qualifying result to canonical driver IDs.

    A car number must resolve through the session's own driver list to exactly one
    Jolpica code; an unmapped or duplicate driver fails rather than being guessed.
    """
    require_utc(available_at, "sprint qualifying availability")
    by_number: dict[int, str] = {}
    for driver in drivers:
        code = driver.get("name_acronym")
        if code not in codes:
            raise ValueError(f"OpenF1 driver {code} has no Jolpica code this season")
        number = int(driver["driver_number"])
        if by_number.get(number, codes[code]) != codes[code]:
            raise ValueError(f"OpenF1 car {number} maps to two drivers")
        by_number[number] = codes[code]
    positions: dict[str, int | None] = {}
    seconds: dict[str, float | None] = {}
    for row in results:
        number = int(row["driver_number"])
        if number not in by_number:
            raise ValueError(f"OpenF1 sprint qualifying car {number} is not in the driver list")
        driver_id = by_number[number]
        if driver_id in positions:
            raise ValueError(f"duplicate OpenF1 sprint qualifying driver {driver_id}")
        position = row.get("position")
        positions[driver_id] = int(position) if position is not None else None
        durations = row.get("duration")
        stages = durations if isinstance(durations, list) else [durations]
        seconds[driver_id] = next(
            (float(value) for value in reversed(stages) if value is not None and value > 0),
            None,
        )
    if not positions:
        raise ValueError("OpenF1 sprint qualifying result is empty")
    # A driver who set no classified time is absent from the result but still entered:
    # keep them with a missing sprint qualifying position rather than dropping them.
    for entrant in by_number.values():
        positions.setdefault(entrant, None)
        seconds.setdefault(entrant, None)
    return SprintGrid(positions, seconds, available_at, source)


def _labels(
    path: Path, available_at: datetime, identities: Crosswalk
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Sprint labels from the normalized Jolpica classification, in canonical IDs."""
    roster: dict[str, str | None] = {}
    labels: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        driver = identities.driver(row["driver_id"])
        text = (row["position_text"] or "").strip()
        classified = text.isdecimal()
        status = (row["status"] or "").strip().lower()
        if status in {"did not start", "disqualified", "withdrew"} or text in {"W", "D", "E"}:
            dnf: bool | None = None
        else:
            dnf = not (status == "finished" or status.startswith("+") or status == "lapped")
        position = row["position"] if classified else None
        roster[driver] = identities.constructor(row["constructor_id"])
        labels[driver] = {
            "label_position": position,
            "label_winner": position == 1,
            "label_podium": position is not None and position <= 3,
            "label_dnf": dnf,
            "label_available_at": available_at,
            "label_dnf_available_at": available_at if dnf is not None else None,
            "points": row["points"],
        }
    return roster, labels


def _cross_check(
    event: EventId, labels: dict[str, dict[str, Any]], evidence: dict[tuple[int, int, str], Any]
) -> str | None:
    """Return an exclusion reason when Jolpica disagrees with the audited FIA sprint."""
    for driver, label in labels.items():
        record = evidence.get((event.season, event.round, driver))
        if record is None:
            return f"no_fia_sprint_record:{driver}"
        fia_points = record.get("sprint_points")
        if fia_points is not None and float(fia_points) != float(label["points"] or 0.0):
            return f"sprint_points_disagree:{driver}"
        fia_position = record.get("sprint_position")
        if (
            fia_position is not None
            and str(fia_position).isdecimal()
            and label["label_position"] is not None
            and int(fia_position) != label["label_position"]
        ):
            return f"sprint_position_disagree:{driver}"
    return None


def _evidence(root: Path) -> dict[tuple[int, int, str], Any]:
    payload = json.loads(
        (root / "data/audit/event_points_evidence.json").read_text(encoding="utf-8")
    )
    return {
        (record["season"], record["round"], record["driver"]): record
        for record in payload["records"]
    }


def _session_window(history: AuditedHistory, event: EventId) -> tuple[datetime, datetime]:
    weekend = history.weekends[event]
    start = weekend.first_practice or weekend.sprint_qualifying or weekend.qualifying
    if start is None or weekend.sprint is None:
        raise ValueError(f"{event.partition()} has no sprint schedule")
    return start - timedelta(hours=12), weekend.sprint


def historical_sprints(
    root: Path,
    history: AuditedHistory,
    *,
    clock: datetime,
    http_client: httpx.Client | None = None,
) -> tuple[list[SprintEvent], list[dict[str, str]]]:
    """Assemble every completed sprint with Development sources and stated exclusions."""
    events: list[SprintEvent] = []
    excluded: list[dict[str, str]] = []
    evidence = _evidence(root)
    seasons = sorted({event.season for event in history.weekends})
    with JsonSourceClient(_LIMITS, http_client) as client:
        identities = {
            season: crosswalk(
                _retained(
                    root,
                    f"season={season}/jolpica-drivers",
                    f"{BASE_URL}/{season}/drivers/",
                    {"limit": 100, "offset": 0},
                    client,
                )
            )
            for season in seasons
        }
        sessions: dict[int, list[dict[str, Any]]] = {}
        for season in seasons:
            if season >= 2023:
                payload = _retained(
                    root,
                    f"season={season}/openf1-sessions",
                    f"{OPENF1}/sessions",
                    {"year": season},
                    client,
                )
                sessions[season] = [
                    item for item in payload if item.get("session_name") in SPRINT_SESSION_NAMES
                ]
        for event, weekend in sorted(history.weekends.items(), key=lambda item: item[0]):
            if weekend.sprint is None or weekend.sprint + SPRINT_LABEL_BOUND > clock:
                continue
            name = event.partition()
            sprint_path = (
                root
                / "data/normalized"
                / f"season={event.season}"
                / f"round={event.round:02d}"
                / "sprint.parquet"
            )
            if not sprint_path.exists():
                excluded.append({"event_id": name, "reason": "jolpica_sprint_missing"})
                continue
            label_time = weekend.sprint + SPRINT_LABEL_BOUND
            roster, labels = _labels(sprint_path, label_time, identities[event.season])
            if any(team is None for team in roster.values()):
                excluded.append({"event_id": name, "reason": "sprint_constructor_missing"})
                continue
            reason = _cross_check(event, labels, evidence)
            if reason is not None:
                excluded.append({"event_id": name, "reason": reason})
                continue
            try:
                if event.season < 2023:
                    grid = _legacy_grid(root, event, weekend.qualifying, identities[event.season])
                else:
                    low, high = _session_window(history, event)
                    matches = [
                        item
                        for item in sessions.get(event.season, [])
                        if low <= datetime.fromisoformat(item["date_start"]) < high
                    ]
                    if len(matches) != 1:
                        raise ValueError("openf1_sprint_qualifying_session_missing_or_ambiguous")
                    session = matches[0]
                    key = int(session["session_key"])
                    results = _retained(
                        root,
                        f"season={event.season}/openf1-session-result-{key}",
                        f"{OPENF1}/session_result",
                        {"session_key": key},
                        client,
                    )
                    drivers = _retained(
                        root,
                        f"season={event.season}/openf1-drivers-{key}",
                        f"{OPENF1}/drivers",
                        {"session_key": key},
                        client,
                    )
                    grid = openf1_grid(
                        results,
                        drivers,
                        identities[event.season].codes,
                        datetime.fromisoformat(session["date_end"]),
                        f"openf1:session_result:{key}",
                    )
            except (ValueError, KeyError) as exc:
                excluded.append({"event_id": name, "reason": str(exc)})
                continue
            if set(grid.positions) != set(roster):
                excluded.append({"event_id": name, "reason": "sprint_grid_roster_mismatch"})
                continue
            if grid.available_at >= weekend.sprint:
                excluded.append({"event_id": name, "reason": "sprint_grid_after_sprint_start"})
                continue
            events.append(
                SprintEvent(
                    event,
                    weekend.circuit_id,
                    grid.available_at,
                    weekend.sprint,
                    {driver: str(team) for driver, team in roster.items()},
                    grid,
                    labels,
                )
            )
    events.sort(key=lambda item: item.cutoff)
    return events, excluded


def _legacy_grid(
    root: Path, event: EventId, qualifying: datetime | None, identities: Crosswalk
) -> SprintGrid:
    """2022: Friday qualifying set the sprint grid."""
    if qualifying is None:
        raise ValueError("legacy_qualifying_schedule_missing")
    path = (
        root
        / "data/normalized"
        / f"season={event.season}"
        / f"round={event.round:02d}"
        / "qualifying.parquet"
    )
    if not path.exists():
        raise ValueError("legacy_qualifying_missing")
    positions: dict[str, int | None] = {}
    seconds: dict[str, float | None] = {}
    for row in pq.read_table(path).to_pylist():
        driver = identities.driver(row["driver_id"])
        positions[driver] = row["position"]
        seconds[driver] = next(
            (row[name] for name in ("q3_seconds", "q2_seconds", "q1_seconds") if row[name]),
            None,
        )
    return SprintGrid(
        positions, seconds, qualifying + LEGACY_GRID_BOUND, "jolpica:qualifying:legacy_2022"
    )


def status_sources(
    root: Path, history: AuditedHistory, *, http_client: httpx.Client | None = None
) -> Callable[[EventId], tuple[dict[str, Any], dict[str, Any] | None]]:
    """Jolpica and OpenF1 sprint status rows by canonical driver, for DNF agreement.

    OpenF1 starts in 2023, so earlier sprints have no OpenF1 rows and their
    retirements stay unlabelled under the three-source rule.
    """
    client = JsonSourceClient(_LIMITS, http_client)

    def provide(event: EventId) -> tuple[dict[str, Any], dict[str, Any] | None]:
        identities = crosswalk(
            _retained(
                root,
                f"season={event.season}/jolpica-drivers",
                f"{BASE_URL}/{event.season}/drivers/",
                {"limit": 100, "offset": 0},
                client,
            )
        )
        path = (
            root
            / "data/normalized"
            / f"season={event.season}"
            / f"round={event.round:02d}"
            / "sprint.parquet"
        )
        if not path.exists():
            raise ValueError("jolpica_sprint_missing")
        jolpica = {
            identities.driver(row["driver_id"]): row for row in pq.read_table(path).to_pylist()
        }
        if event.season < 2023:
            return jolpica, None
        sessions = _retained(
            root,
            f"season={event.season}/openf1-sprint-sessions",
            f"{OPENF1}/sessions",
            {"year": event.season, "session_name": "Sprint"},
            client,
        )
        low, high = _session_window(history, event)
        matches = [
            item
            for item in sessions
            if low <= datetime.fromisoformat(item["date_start"]) <= high + timedelta(hours=2)
        ]
        if len(matches) != 1:
            return jolpica, None
        key = int(matches[0]["session_key"])
        results = _retained(
            root,
            f"season={event.season}/openf1-session-result-{key}",
            f"{OPENF1}/session_result",
            {"session_key": key},
            client,
        )
        drivers = _retained(
            root,
            f"season={event.season}/openf1-drivers-{key}",
            f"{OPENF1}/drivers",
            {"session_key": key},
            client,
        )
        by_number = {
            int(item["driver_number"]): identities.codes.get(item["name_acronym"])
            for item in drivers
        }
        openf1 = {
            str(by_number[int(row["driver_number"])]): row
            for row in results
            if by_number.get(int(row["driver_number"])) is not None
        }
        return jolpica, openf1

    return provide


def latest_jolpica_constructors(root: Path, event: EventId, drivers: list[str]) -> dict[str, str]:
    """Constructor of each driver at their latest Jolpica entry earlier this season.

    Used only for a driver missing from the latest audited roster (a mid-season
    change). Identifiers are mapped to canonical Gold IDs through the retained
    season driver list when one exists.
    """
    if not drivers:
        return {}
    retained = root / _RAW / f"season={event.season}" / "jolpica-drivers.json"
    identities = (
        crosswalk(json.loads(retained.read_text(encoding="utf-8"))["payload"])
        if retained.exists()
        else Crosswalk({}, {})
    )
    found: dict[str, str] = {}
    for number in range(event.round - 1, 0, -1):
        base = root / "data/normalized" / f"season={event.season}" / f"round={number:02d}"
        for name in ("results.parquet", "sprint.parquet", "qualifying.parquet"):
            path = base / name
            if not path.exists():
                continue
            for row in pq.read_table(path, columns=["driver_id", "constructor_id"]).to_pylist():
                driver = identities.driver(row["driver_id"])
                team = Crosswalk.constructor(row["constructor_id"])
                if driver in drivers and driver not in found and team is not None:
                    found[driver] = team
        if set(found) == set(drivers):
            break
    return found


def gold_sprints(
    root: Path, history: AuditedHistory, *, clock: datetime
) -> tuple[list[SprintEvent], list[dict[str, str]], dict[str, Any]] | None:
    """Completed sprints from the current Gold sprint version, or None if none exists."""
    loaded = load_gold_sprints(root)
    if loaded is None:
        return None
    rows, manifest, digest = loaded
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["event_id"], []).append(row)
    events = []
    for event_id, group in grouped.items():
        season, number = event_id.split("/")
        event = EventId(int(season.removeprefix("season=")), int(number.removeprefix("round=")))
        weekend = history.weekends.get(event)
        labels_at = group[0]["label_available_at"]
        if weekend is None or weekend.sprint is None or labels_at > clock:
            continue
        grid = SprintGrid(
            {row["driver_id"]: row["sprint_qualifying_position"] for row in group},
            {row["driver_id"]: row["sprint_qualifying_last_seconds"] for row in group},
            group[0]["grid_available_at"],
            f"fia:{SPRINT_GOLD_VERSION}:{digest[:12]}",
        )
        labels = {
            row["driver_id"]: {
                "label_position": row["label_position"],
                "label_winner": row["label_position"] == 1,
                "label_podium": row["label_position"] is not None and row["label_position"] <= 3,
                "label_dnf": row["label_dnf"],
                "label_available_at": row["label_available_at"],
                "label_dnf_available_at": row["label_available_at"]
                if row["label_dnf"] is not None
                else None,
                "points": None,
            }
            for row in group
        }
        events.append(
            SprintEvent(
                event,
                weekend.circuit_id,
                grid.available_at,
                weekend.sprint,
                {row["driver_id"]: row["constructor_id"] for row in group},
                grid,
                labels,
            )
        )
    events.sort(key=lambda item: item.cutoff)
    return events, list(manifest["excluded"]), {"manifest_sha256": digest, **manifest["dataset"]}


def weekend_values(grid: SprintGrid, roster: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Sprint qualifying values in the race contract's qualifying fields."""
    values: dict[str, dict[str, Any]] = {}
    for driver, constructor in roster.items():
        position = grid.positions.get(driver)
        mates = [
            grid.positions.get(other)
            for other, team in roster.items()
            if team == constructor and other != driver
        ]
        mate = mates[0] if len(mates) == 1 else None
        values[driver] = {
            "qualifying_position": position,
            "qualifying_last_session_seconds": grid.last_seconds.get(driver),
            "teammate_qualifying_position_delta": position - mate
            if position is not None and mate is not None
            else None,
        }
        for name in tuple(values[driver]):
            values[driver][f"{name}_available_at"] = grid.available_at
    return values


def sprint_rows(
    history: AuditedHistory,
    sprint: SprintEvent,
    *,
    labelled: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label-free race-contract rows for a sprint roster, optionally with sprint labels."""
    rows, reasons = build_rows(
        history,
        RACE_CONTRACT,
        event=sprint.event,
        circuit_id=sprint.circuit_id,
        cutoff=sprint.cutoff,
        roster=sprint.roster,
        weekend=weekend_values(sprint.grid, sprint.roster),
    )
    for row in rows:
        row["cutoff_kind"] = SPRINT_CONTRACT
        if labelled:
            label = sprint.labels[row["driver_id"]]
            row.update({key: value for key, value in label.items() if key != "points"})
    return rows, reasons


def _sprint_prior_rows(sprints: list[SprintEvent], cutoff: datetime) -> list[dict[str, Any]]:
    """Earlier sprint retirement labels known by the cutoff."""
    return [
        {"label_dnf": label["label_dnf"]}
        for sprint in sprints
        for label in sprint.labels.values()
        if sprint.sprint_start < cutoff
        and label["label_dnf"] is not None
        and label["label_dnf_available_at"] <= cutoff
    ]


def collect_sprint_records(
    history: AuditedHistory,
    race_rows: list[dict[str, Any]],
    sprints: list[SprintEvent],
    *,
    seed: int = 42,
    candidates: tuple[str, ...] = JOINT_CANDIDATES,
    progress: Callable[[str], None] | None = None,
) -> list[RaceRecord]:
    """Out-of-fold sprint records: race-trained candidates applied to sprint rows."""
    columns, dnf_cols = columns_for(RACE_CONTRACT, MASKED)
    records = []
    for sprint in sprints:
        rows, _ = sprint_rows(history, sprint)
        test = sorted(rows, key=lambda row: row["driver_id"])
        train, events = training_rows(race_rows, sprint.cutoff)
        if events < MIN_TRAIN_EVENTS:
            continue
        for row in train:
            require_known_by(row["label_available_at"], sprint.cutoff)
        winner = [index for index, row in enumerate(test) if row["label_winner"]]
        if len(winner) != 1:
            continue
        record = RaceRecord(
            sprint.event.partition(),
            sprint.cutoff,
            [row["driver_id"] for row in test],
            winner[0],
            np.asarray([float(row["label_podium"]) for row in test]),
            np.asarray(
                [np.nan if row["label_position"] is None else row["label_position"] for row in test]
            ),
            [row["label_dnf"] for row in test],
            bool(test[0]["circuit_seen_before"]),
            events,
        )
        for name in candidates:
            model = fit_strength(name, train, columns, seed=seed)
            record.utilities[name] = model.utility(test)
            record.devices[name] = model.device
        record.baselines = baseline_marginals(train, test, columns, seed)
        prior = fit_dnf("prior", _sprint_prior_rows(sprints, sprint.cutoff), (), seed)
        record.dnf["prior"] = prior.predict(test)
        for name in DNF_CANDIDATES:
            if name != "prior":
                record.dnf[name] = fit_dnf(name, train, dnf_cols, seed).predict(test)
        records.append(record)
        if progress is not None:
            progress(f"sprint {sprint.event.partition()} ({len(records)} outer sprints)")
    return records


def evaluate_sprints(
    root: Path,
    history: AuditedHistory,
    race_rows: list[dict[str, Any]],
    race_dataset_sha256: str,
    sprints: list[SprintEvent],
    excluded: list[dict[str, str]],
    *,
    seed: int = 42,
    candidates: tuple[str, ...] = JOINT_CANDIDATES,
    progress: Callable[[str], None] | None = None,
    tier: str = "Development",
) -> tuple[dict[str, Any], Path]:
    """Run or reload the sprint evaluation (addendum by tier) for exactly these inputs."""
    addendum, addendum_sha256 = addendum_for(tier)
    sprint_identity = [
        {
            "event_id": sprint.event.partition(),
            "cutoff": sprint.cutoff.isoformat(),
            "grid_source": sprint.grid.source,
            "labels": {
                driver: [label["label_position"], label["label_dnf"]]
                for driver, label in sorted(sprint.labels.items())
            },
            "grid": [
                [driver, position] for driver, position in sorted(sprint.grid.positions.items())
            ],
        }
        for sprint in sprints
    ]
    key_payload = {
        "addendum_sha256": addendum_sha256,
        "race_dataset_sha256": race_dataset_sha256,
        "sprints_sha256": hashlib.sha256(_canonical(sprint_identity)).hexdigest(),
        "seed": seed,
        "candidates": list(candidates),
    }
    key = hashlib.sha256(_canonical(key_payload)).hexdigest()
    path = (
        root
        / "models/experiments/gold"
        / history.version.dataset_version
        / addendum["version"].replace("-", "_")
        / key
        / "evaluation.json"
    )
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("key") != key_payload:
            raise ValueError("cached sprint evaluation does not match its inputs")
        return cached, path
    oof_path = path.parent / "oof.joblib"
    if oof_path.exists():
        records = joblib.load(oof_path)
    else:
        records = collect_sprint_records(
            history, race_rows, sprints, seed=seed, candidates=candidates, progress=progress
        )
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(records, oof_path)
    report = analyse(records, seed=seed, candidates=candidates)
    report["addendum"] = addendum
    report["addendum_sha256"] = addendum_sha256
    report["evidence_tier"] = tier
    report["sprints"] = sprint_identity
    report["excluded_sprints"] = excluded
    report["key"] = key_payload
    report["gold_dataset_version"] = history.version.dataset_version
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode())
    return report, path


class SprintModel:
    """Live sprint models: race-trained candidates with sprint calibration.

    It exposes the same interface as ``ContractModel`` (``components``, ``dnf``,
    ``distribution``) so the joint sampler and season simulation treat it alike.
    """

    contract = SPRINT_CONTRACT

    def __init__(
        self,
        evaluation: dict[str, Any],
        race_rows: list[dict[str, Any]],
        sprints: list[SprintEvent],
        *,
        cutoff: datetime,
        seed: int,
        candidates: tuple[str, ...] = JOINT_CANDIDATES,
        evaluation_path: Path | None = None,
        dataset_sha256: str = "",
    ) -> None:
        self.evaluation = evaluation
        # The sprint evaluation is both the protocol run and the live-masked run.
        self.protocol = evaluation
        self.protocol_path = evaluation_path
        self.evaluation_path = evaluation_path
        self.dataset_sha256 = dataset_sha256
        self.candidates = candidates
        self.masked = MASKED
        self.columns, self.dnf_columns = columns_for(RACE_CONTRACT, MASKED)
        self.training, self.training_events = training_rows(race_rows, cutoff)
        for row in self.training:
            require_known_by(row["label_available_at"], cutoff)
        live = evaluation["live"]
        self.primary: str = live["primary"]
        self.members: list[str] = live["primary_members"]
        self.models: dict[str, StrengthModel] = {
            name: fit_strength(name, self.training, self.columns, seed=seed) for name in candidates
        }
        self.dnf_name: str = live["dnf_model"]
        self.dnf = (
            fit_dnf("prior", _sprint_prior_rows(sprints, cutoff), (), seed)
            if self.dnf_name == "prior"
            else fit_dnf(self.dnf_name, self.training, self.dnf_columns, seed)
        )
        self.sprint_labels = sum(len(_sprint_prior_rows([sprint], cutoff)) for sprint in sprints)
        self.seed = seed

    def calibration(self, name: str, unseen: bool) -> tuple[float, float, str]:
        live = self.evaluation["live"]
        rule = live["unseen_calibration"].get(name) if unseen else None
        source = rule or live["calibration"][name]
        return source["temperature"], source["shrink"], "unseen_rule" if rule else "shared"

    def components(
        self,
        rows: list[dict[str, Any]],
        unseen: bool,
        *,
        members: list[str] | None = None,
        calibrated: bool = True,
        allow_unseen_rule: bool = True,
        temperature_scale: float = 1.0,
    ) -> list[tuple[np.ndarray[Any, Any], float, float]]:
        result = []
        for name in members or self.members:
            tau, shrink, _ = self.calibration(name, unseen and allow_unseen_rule)
            if not calibrated:
                tau, shrink = 1.0, 0.0
            result.append((self.models[name].utility(rows), tau * temperature_scale, shrink))
        return result

    def distribution(
        self,
        rows: list[dict[str, Any]],
        components: list[Any],
        *,
        draws: int,
        seed: int,
        clean: bool = False,
    ) -> dict[str, Any]:
        dnf = self.dnf.predict(rows)
        order, retired = sample_mixture(
            components, dnf, draws=draws, rng=np.random.default_rng(seed)
        )
        values = marginals(order, retired)
        values["dnf_model"] = dnf
        if clean:
            clean_order, clean_retired = sample_mixture(
                components, np.zeros_like(dnf), draws=draws, rng=np.random.default_rng(seed)
            )
            clean_values = marginals(clean_order, clean_retired)
            values["clean_finish"] = clean_values["finish"]
            values["clean_expected"] = clean_values["expected"]
        return values
