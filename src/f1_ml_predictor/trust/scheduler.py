"""Bounded prospective ticks with immutable captures and audited label imports.

The schedule names an earliest polling time, never a qualifying completion time.
A validated nonempty response establishes an observation-time decision. Separate
normalization verifies frozen JSON before feature registration. Audited outcomes
remain separate from the predictive capture.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import normalize_qualifying, normalize_schedule
from f1_ml_predictor.sources.http import JsonSourceClient, SourceError
from f1_ml_predictor.sources.jolpica import BASE_URL
from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.collected_features import attach_audited_outcomes, certify_capture
from f1_ml_predictor.trust.collected_outcomes import collect_final_outcomes
from f1_ml_predictor.trust.collector import collect_weekend
from f1_ml_predictor.trust.locking import advisory_lock
from f1_ml_predictor.trust.outcomes import OUTCOME_SCHEMA, validate_audited_outcomes
from f1_ml_predictor.trust.prospective import load_bundle

_LIMITS = ((4, 1.0), (500, 3600.0))
_DIRECTORY = Path("data/raw/prospective_scheduler")


def _clock(now: Callable[[], datetime] | None) -> datetime:
    timestamp = datetime.now(UTC) if now is None else now()
    if not isinstance(timestamp, datetime):
        raise ValueError("scheduler clock must return a timezone-aware UTC datetime")
    require_utc(timestamp, "scheduler clock")
    return timestamp


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _member(root: Path, name: str) -> Path:
    path = Path(name)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("scheduler member escapes project root")
    return resolved


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical(value))


def _save(root: Path, state: dict[str, Any]) -> None:
    path = root / _DIRECTORY / "status.json"
    temporary = path.with_name(f".status-{uuid4().hex}.json")
    try:
        _write_new(temporary, state)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    """Exclude concurrent writers without leaving stale ownership after exit."""
    with advisory_lock(root / _DIRECTORY / ".tick.lock"):
        yield


def scheduler_status(root: Path) -> dict[str, Any]:
    """Read persisted status without network requests or eligibility promotion."""
    path = root / _DIRECTORY / "status.json"
    if not path.exists():
        return {"version": 1, "status": "idle", "events": {}, "evaluation_eligible": False}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != 1
            or not isinstance(state.get("events"), dict)
            or type(state.get("evaluation_eligible")) is not bool
        ):
            raise ValueError("invalid scheduler status")
        for name, event in state["events"].items():
            if not isinstance(event, dict) or not isinstance(event.get("captures"), list):
                raise ValueError("invalid scheduler event status")
            identity = EventId(event["season"], event["round"])
            if identity.partition() != name:
                raise ValueError("scheduler event identity mismatch")
            require_utc(datetime.fromisoformat(event["race_start"]), "scheduled race start")
        return dict(state)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid scheduler status") from exc


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("invalid Jolpica pagination count")
    if isinstance(value, str) and not value.isdecimal():
        raise ValueError("invalid Jolpica pagination count")
    count = int(value)
    if count < 0:
        raise ValueError("invalid Jolpica pagination count")
    return count


def _races(payload: Any) -> tuple[list[dict[str, Any]], int]:
    try:
        mrdata = payload["MRData"]
        races = mrdata["RaceTable"]["Races"]
        total = _count(mrdata["total"])
        if (
            _count(mrdata["offset"]) != 0
            or not 1 <= _count(mrdata["limit"]) <= 100
            or total > _count(mrdata["limit"])
            or not isinstance(races, list)
            or not all(isinstance(race, dict) for race in races)
        ):
            raise ValueError("incomplete or malformed bounded Jolpica response")
        return races, total
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed Jolpica response") from exc


def _observe(
    root: Path,
    source: JsonSourceClient,
    url: str,
    role: str,
    now: Callable[[], datetime] | None,
) -> tuple[Any, datetime, str]:
    params: dict[str, str | int | float] = {"limit": 100, "offset": 0}
    payload = source.get_json(url, params)
    captured = _clock(now)
    name = _DIRECTORY / "observations" / f"{role}-{uuid4().hex}.json"
    _write_new(
        root / name,
        {
            "version": 1,
            "provider": "jolpica",
            "url": url,
            "params": params,
            "role": role,
            "response_captured_at": captured.isoformat(),
            "payload_sha256": _digest(_canonical(payload)),
            "payload": payload,
        },
    )
    return payload, captured, name.as_posix()


def _qualifying_roster(bundle: Path) -> tuple[dict[str, Any], set[str]]:
    manifest, tables = load_bundle(bundle)
    event = EventId(**manifest["event"])
    requests = manifest["request_metadata"].get("source_requests", {})
    if not isinstance(requests, dict):
        raise ValueError("capture has invalid source request metadata")
    rosters: list[set[str]] = []
    for name, request in requests.items():
        if not isinstance(request, dict):
            raise ValueError("capture has invalid source request metadata")
        if request.get("role") != "qualifying":
            continue
        if name not in tables or tables[name].column_names != ["payload_json"]:
            raise ValueError("capture is missing raw qualifying payload")
        if tables[name].num_rows != 1:
            raise ValueError("capture requires one raw qualifying payload")
        payload = json.loads(tables[name]["payload_json"][0].as_py())
        races, total = _races(payload)
        qualifying = normalize_qualifying(races, event)
        if qualifying.num_rows != total or not total or len(races) != 1:
            raise ValueError("capture has incomplete qualifying roster")
        rosters.append(set(qualifying["driver_id"].to_pylist()))
    if len(rosters) != 1:
        raise ValueError("capture requires one validated qualifying roster")
    return manifest, rosters[0]


def _capture_record(root: Path, bundle: Path, version: int, plan: str | None) -> dict[str, Any]:
    manifest, roster = _qualifying_roster(bundle)
    _, tables = load_bundle(bundle)
    sources = {}
    for name, request in manifest["request_metadata"]["source_requests"].items():
        payload = json.loads(tables[name]["payload_json"][0].as_py())
        sources[name] = {
            **request,
            "payload_sha256": _digest(_canonical(payload)),
        }
    certification = certify_capture(root, bundle)
    return {
        "capture_version": version,
        "bundle": bundle.relative_to(root).as_posix(),
        "manifest_sha256": manifest["manifest_sha256"],
        "captured_at": manifest["captured_at"],
        "cutoff_kind": manifest["cutoff_kind"],
        "plan": plan,
        "roster": sorted(roster),
        "sources": sources,
        **certification,
    }


def _reconcile(root: Path, entry: dict[str, Any]) -> None:
    """Recover verified orphan captures after interruption, without recapturing."""
    event = EventId(entry["season"], entry["round"])
    directory = root / "data/raw/prospective" / event.partition()
    known = {capture["bundle"] for capture in entry["captures"]}
    for capture in entry["captures"]:
        bundle = _member(root, capture["bundle"])
        manifest, _ = _qualifying_roster(bundle)
        if manifest["manifest_sha256"] != capture["manifest_sha256"]:
            raise ValueError("recorded capture manifest hash changed")
        capture.update(certify_capture(root, bundle))
    for kind in ("post_qualifying", "pre_race"):
        for bundle in sorted((directory / kind).glob("*")):
            if not bundle.is_dir() or bundle.name.startswith("."):
                continue
            relative = bundle.relative_to(root).as_posix()
            if relative in known:
                continue
            manifest, _ = _qualifying_roster(bundle)
            if (
                manifest["request_metadata"].get("window", {}).get("race_start")
                != entry["race_start"]
            ):
                continue
            entry["captures"].append(
                _capture_record(root, bundle, len(entry["captures"]) + 1, None)
            )
            known.add(relative)


def _tick(
    root: Path,
    state: dict[str, Any],
    *,
    season: int,
    now: Callable[[], datetime] | None,
    http_client: httpx.Client | None,
    collect_pre_race: bool,
    pre_race_minutes: int,
    new_capture: bool,
    forecast_request: dict[str, Any] | None,
) -> None:
    clock = _clock(now)
    for entry in state["events"].values():
        _reconcile(root, entry)
        if datetime.fromisoformat(entry["race_start"]) <= clock:
            entry["status"] = "closed" if entry["captures"] else "missed"
    with JsonSourceClient(_LIMITS, http_client) as source:
        schedule_url = f"{BASE_URL}/{season}/"
        payload, observed, observation = _observe(root, source, schedule_url, "schedule", now)
        races, total = _races(payload)
        if len(races) != total:
            raise ValueError("schedule response is incomplete")
        schedule = normalize_schedule(races, season)
        state["schedule_observation"] = observation
        upcoming = [
            row
            for row in schedule.to_pylist()
            if row["race_start_utc"] is not None and row["race_start_utc"] > observed
        ]
        if not upcoming:
            state["status"] = "no_upcoming_race"
            return
        selected = min(upcoming, key=lambda row: row["race_start_utc"])
        event = EventId(selected["season"], selected["round"])
        race_start = selected["race_start_utc"]
        if race_start.year != season:
            raise ValueError("scheduled race date disagrees with season")
        raw = next(race for race in races if int(race["round"]) == event.round)
        name = event.partition()
        entry = state["events"].setdefault(
            name,
            {
                "season": event.season,
                "round": event.round,
                "race_name": selected["race_name"],
                "race_start": race_start.isoformat(),
                "captures": [],
                "status": "waiting_for_qualifying",
            },
        )
        state["active_event"] = name
        if entry["race_start"] != race_start.isoformat():
            entry["status"] = state["status"] = "schedule_changed_review_required"
            return
        _reconcile(root, entry)
        qualifying = raw.get("Qualifying")
        if not isinstance(qualifying, dict) or not qualifying.get("time"):
            entry["status"] = state["status"] = "qualifying_schedule_missing"
            return
        scheduled = datetime.fromisoformat(f"{qualifying['date']}T{qualifying['time']}")
        require_utc(scheduled, "scheduled qualifying start")
        if scheduled >= race_start or scheduled.year != season:
            raise ValueError("invalid scheduled qualifying start")
        entry["qualifying_scheduled_start"] = scheduled.isoformat()
        captured_kinds = {capture["cutoff_kind"] for capture in entry["captures"]}
        kind = "post_qualifying"
        if "post_qualifying" in captured_kinds and not new_capture:
            if (
                not collect_pre_race
                or "pre_race" in captured_kinds
                or observed < race_start - timedelta(minutes=pre_race_minutes)
            ):
                entry["status"] = state["status"] = "captured"
                return
            kind = "pre_race"
        if observed < scheduled:
            entry["status"] = state["status"] = "waiting_for_qualifying"
            return
        qualifying_url = f"{BASE_URL}/{season}/{event.round}/qualifying/"
        payload, decision, witness = _observe(root, source, qualifying_url, "qualifying", now)
        qualifying_races, total = _races(payload)
        normalized = normalize_qualifying(qualifying_races, event)
        if not total and not normalized.num_rows:
            entry["status"] = state["status"] = "waiting_for_qualifying_results"
            return
        if len(qualifying_races) != 1 or total != normalized.num_rows:
            raise ValueError("qualifying response is incomplete")
        if decision >= race_start:
            entry["status"] = state["status"] = "missed"
            return
        entry["qualifying_decision_at"] = decision.isoformat()
        entry["qualifying_decision_basis"] = "fresh_nonempty_qualifying_response_observed"
        entry["qualifying_observation"] = witness
        version = len(entry["captures"]) + 1
        plans = root / _DIRECTORY / "plans" / name / kind
        existing = list(plans.glob("*.json"))
        plan_name = _DIRECTORY / "plans" / name / kind / f"v{len(existing) + 1:04d}.json"
        requests = [
            {
                "name": "schedule",
                "role": "event",
                "url": schedule_url,
                "params": {"limit": 100, "offset": 0},
            },
            {
                "name": "qualifying",
                "role": "qualifying",
                "url": qualifying_url,
                "params": {"limit": 100, "offset": 0},
            },
        ]
        if forecast_request is not None:
            if forecast_request.get("role") != "forecast":
                raise ValueError("optional weather request must have forecast role")
            requests.append(forecast_request)
        plan = {
            "version": 1,
            "season": season,
            "round": event.round,
            "race_start": race_start.isoformat(),
            "qualifying_decision_at": decision.isoformat(),
            "qualifying_decision_basis": entry["qualifying_decision_basis"],
            "qualifying_observation": witness,
            "capture_version": version,
            "cutoff_kind": kind,
            "pre_race_minutes": pre_race_minutes,
            "requests": requests,
        }
        _write_new(root / plan_name, plan)
        entry["pending_plan"] = plan_name.as_posix()
        _save(root, state)
        bundle = collect_weekend(root / plan_name, root, http_client=http_client)
        if bundle.relative_to(root).as_posix() not in {
            capture["bundle"] for capture in entry["captures"]
        }:
            entry["captures"].append(_capture_record(root, bundle, version, plan_name.as_posix()))
        entry.pop("pending_plan", None)
        entry["status"] = state["status"] = "captured"


def scheduler_tick(
    root: Path,
    *,
    season: int | None = None,
    collect_pre_race: bool = False,
    pre_race_minutes: int = 60,
    new_capture: bool = False,
    forecast_request: dict[str, Any] | None = None,
    min_interval_seconds: int = 300,
    http_client: httpx.Client | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Perform one bounded tick; further capture versions require explicit intent.

    ``now`` controls scheduling only. The collector still enforces its own live
    clock and fresh capture checks; this cannot manufacture a historical capture.
    Network/validation failures persist an error and can retry on the next tick.
    """
    root = root.resolve()
    clock = _clock(now)
    selected_season = clock.year if season is None else season
    EventId(selected_season, 1)
    if type(min_interval_seconds) is not int or not 60 <= min_interval_seconds <= 3600:
        raise ValueError("scheduler interval must be between 60 and 3600 seconds")
    if type(pre_race_minutes) is not int or not 1 <= pre_race_minutes <= 1440:
        raise ValueError("pre-race window must be between 1 and 1440 minutes")
    if selected_season != clock.year and selected_season != clock.year + 1:
        raise ValueError("automatic collection only supports the current or next season")
    with _lock(root):
        state = scheduler_status(root)
        due = state.get("next_check_at")
        if due is not None and clock < datetime.fromisoformat(due) and not new_capture:
            return state
        state["last_tick_at"] = clock.isoformat()
        state["next_check_at"] = (clock + timedelta(seconds=min_interval_seconds)).isoformat()
        state.pop("error", None)
        try:
            _tick(
                root,
                state,
                season=selected_season,
                now=now,
                http_client=http_client,
                collect_pre_race=collect_pre_race,
                pre_race_minutes=pre_race_minutes,
                new_capture=new_capture,
                forecast_request=forecast_request,
            )
        except (ValueError, SourceError, OSError, KeyError, TypeError, OverflowError) as exc:
            state["status"] = "error"
            state["error"] = str(exc)
        _collect_pending_labels(root, state, http_client=http_client, now=now)
        _save(root, state)
        return state


def _collect_pending_labels(
    root: Path,
    state: dict[str, Any],
    *,
    http_client: httpx.Client | None,
    now: Callable[[], datetime] | None,
) -> None:
    clock = _clock(now)
    missing = [
        entry
        for entry in state["events"].values()
        if entry["captures"]
        and not entry.get("outcome_versions")
        and datetime.fromisoformat(entry["race_start"]) < clock
    ]
    if not missing:
        return
    entry = min(missing, key=lambda item: datetime.fromisoformat(item["race_start"]))
    entry["outcome_collection"] = {"status": "attempting", "attempted_at": clock.isoformat()}
    _save(root, state)
    try:
        collected = collect_final_outcomes(
            root, _member(root, entry["captures"][0]["bundle"]), http_client=http_client, now=now
        )
        imported = _import_scheduler_outcomes_locked(
            root,
            state,
            _member(root, collected["path"]),
            expected_sha256=collected["sha256"],
            source_audit=collected["audit"],
            now=now,
        )
        entry["outcome_collection"].update(
            {"status": "captured", "source": collected, "registered_outcomes": imported}
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        httpx.HTTPError,
    ) as exc:
        entry["outcome_collection"].update({"status": "deferred", "reason": str(exc)})


def import_scheduler_outcomes(
    root: Path,
    outcomes_path: Path,
    *,
    expected_sha256: str,
    field_roster: Collection[str] | None = None,
    now: Callable[[], datetime] | None = None,
    source_audit: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attach later audited labels to verified capture hashes, never to raw inputs."""
    root = root.resolve()
    with _lock(root):
        state = scheduler_status(root)
        record = _import_scheduler_outcomes_locked(
            root,
            state,
            outcomes_path,
            expected_sha256=expected_sha256,
            field_roster=field_roster,
            now=now,
            source_audit=source_audit,
        )
        _save(root, state)
        return record


def _import_scheduler_outcomes_locked(
    root: Path,
    state: dict[str, Any],
    outcomes_path: Path,
    *,
    expected_sha256: str,
    field_roster: Collection[str] | None = None,
    now: Callable[[], datetime] | None = None,
    source_audit: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = outcomes_path.read_bytes()
    digest = _digest(data)
    if digest != expected_sha256:
        raise ValueError("outcome file SHA256 does not match the expected hash")
    table = pq.read_table(pa.BufferReader(data))
    if not table.schema.equals(OUTCOME_SCHEMA, check_metadata=False) or not table.num_rows:
        raise ValueError("import requires a nonempty exact OUTCOME_SCHEMA table")
    identities = {EventId(row["season"], row["round"]) for row in table.to_pylist()}
    if len(identities) != 1:
        raise ValueError("outcome import requires exactly one event")
    event = next(iter(identities))
    clock = _clock(now)
    entry = state["events"].get(event.partition())
    if entry is None or not entry["captures"]:
        raise ValueError("outcomes require a recorded prospective capture")
    race_start = datetime.fromisoformat(entry["race_start"])
    if clock <= race_start:
        raise ValueError("audited outcomes can only be imported after race start")
    capture_hashes: list[str] = []
    roster: set[str] | None = None
    for capture in entry["captures"]:
        manifest, captured_roster = _qualifying_roster(_member(root, capture["bundle"]))
        if EventId(**manifest["event"]) != event:
            raise ValueError("recorded capture belongs to another event")
        if manifest["request_metadata"].get("window", {}).get("race_start") != entry["race_start"]:
            raise ValueError("recorded capture race start disagrees with scheduler event")
        if manifest["manifest_sha256"] != capture["manifest_sha256"]:
            raise ValueError("recorded capture manifest hash changed")
        if roster is not None and captured_roster != roster:
            raise ValueError("capture versions have conflicting field rosters")
        roster = captured_roster
        capture.update(certify_capture(root, _member(root, capture["bundle"])))
        capture_hashes.append(manifest["manifest_sha256"])
    assert roster is not None
    if field_roster is not None and set(field_roster) != roster:
        raise ValueError("supplied field roster disagrees with captured qualifying")
    validate_audited_outcomes(table, field_roster={(event, driver) for driver in roster})
    if any(not race_start < row["label_available_at"] <= clock for row in table.to_pylist()):
        raise ValueError("audited label availability must be after race start and known now")
    if source_audit is not None:
        proof_bytes = _member(root, source_audit["path"]).read_bytes()
        if _digest(proof_bytes) != source_audit["sha256"]:
            raise ValueError("automatic outcome audit manifest hash changed")
        proof = json.loads(proof_bytes)
        if (
            proof.get("version") != 1
            or proof.get("audit_method") != "fia-prospective-final-v1"
            or proof.get("status") != "final"
            or proof["event_id"] != event.partition()
            or proof["capture_manifest_sha256"] not in capture_hashes
            or proof["outcomes"]["sha256"] != digest
            or proof.get("version_audited") is not True
            or proof.get("latest_final_audited") is not True
            or any(row["audit_reference"] != proof["audit_reference"] for row in table.to_pylist())
        ):
            raise ValueError("automatic outcome audit does not bind the exact event/capture/labels")
        available = datetime.fromisoformat(proof["label_available_at_utc"])
        require_utc(available, "automatic outcome observation clock")
        observations = proof["source_observations"]
        if (
            len(observations) != 5
            or observations[2]["url"] != proof["registry_url"]
            or observations[3]["url"] != proof["document_url"]
            or observations[4]["url"] != proof["registry_url"]
            or any(row["label_available_at"] != available for row in table.to_pylist())
        ):
            raise ValueError(
                "automatic outcome audit has inconsistent source/availability bindings"
            )
        for artifact in observations:
            if _digest(_member(root, artifact["path"]).read_bytes()) != artifact["sha256"]:
                raise ValueError("automatic outcome retained source hash changed")
            observed = datetime.fromisoformat(artifact["observed_at_utc"])
            require_utc(observed, "automatic outcome source observation")
            if not race_start < observed <= available:
                raise ValueError("automatic outcome source observation has an invalid clock")
    destination = _DIRECTORY / "outcomes" / event.partition() / f"{digest}.parquet"
    path = root / destination
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("immutable outcome import collision")
    else:
        with path.open("xb") as handle:
            handle.write(data)
    record: dict[str, Any] = {
        "path": destination.as_posix(),
        "sha256": digest,
        "capture_manifest_sha256": sorted(capture_hashes),
        "roster": sorted(roster),
        "evaluation_eligible": False,
        "normalization_required": False,
    }
    if source_audit is not None:
        record["source_audit"] = source_audit
    reference = {"path": destination.as_posix(), "sha256": digest}
    if source_audit is not None:
        reference.update(
            {"audit_path": source_audit["path"], "audit_sha256": source_audit["sha256"]}
        )
    registration = attach_audited_outcomes(root, capture_hashes, reference)
    eligible = set(registration["eligible_capture_manifest_sha256"])
    if eligible != set(capture_hashes):
        raise ValueError("audited outcomes did not produce all registered Gold benchmarks")
    record.update(
        {
            "evaluation_eligible": True,
            "benchmark": registration["benchmark"],
            "registry": registration["registry"],
        }
    )
    for capture in entry["captures"]:
        capture["evaluation_eligible"] = capture["manifest_sha256"] in eligible
    state["evaluation_eligible"] = any(
        capture["evaluation_eligible"]
        for event_entry in state["events"].values()
        for capture in event_entry["captures"]
    )
    outcomes = entry.setdefault("outcome_versions", [])
    if record not in outcomes:
        outcomes.append(record)
    return record
