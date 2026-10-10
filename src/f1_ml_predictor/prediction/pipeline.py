"""Cutoff-specific development predictions for the next race and the season.

The cutoff decides the feature contract. The contract's frozen v3 evaluation
(with the live run's unavailable predictors hidden) decides the primary model,
its calibration and the DNF model. Race outputs come from one coherent joint
sampler. The season simulation samples persistent strength worlds so model
uncertainty is shared across events. Every artifact is ``development_only``.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.models.boosting import BACKENDS
from f1_ml_predictor.models.development import _check_race
from f1_ml_predictor.models.hardware import library_versions
from f1_ml_predictor.prediction.candidates import (
    StrengthModel,
    baseline_marginals,
    fit_dnf,
    fit_strength,
)
from f1_ml_predictor.prediction.contracts import (
    CONTRACT_VERSION,
    GRID_NUMERIC,
    QUALIFYING_NUMERIC,
    SPRINT_NUMERIC,
    AuditedHistory,
    build_contract_datasets,
    build_rows,
    numeric_features,
)
from f1_ml_predictor.prediction.cutoff_report import render_report
from f1_ml_predictor.prediction.drift import estimate_form_drift
from f1_ml_predictor.prediction.evaluation import columns_for, training_rows
from f1_ml_predictor.prediction.joint import marginals, sample_mixture, sharpness
from f1_ml_predictor.prediction.legacy import _capture_base, _latest_roster
from f1_ml_predictor.prediction.live_features import load_schedule
from f1_ml_predictor.prediction.ood import ood_report
from f1_ml_predictor.prediction.protocol import (
    JOINT_CANDIDATES,
    PROTOCOL_V3_SHA256,
    PROTOCOL_V3_VERSION,
)
from f1_ml_predictor.prediction.schedules import weekends
from f1_ml_predictor.prediction.season import (
    ELIGIBILITY_POLICY,
    points_rules,
    published_standings,
    remaining_sessions,
)
from f1_ml_predictor.prediction.sprint import (
    SPRINT_CONTRACT,
    SprintEvent,
    SprintModel,
    addendum_for,
    crosswalk,
    evaluate_sprints,
    gold_sprints,
    historical_sprints,
    latest_jolpica_constructors,
    sprint_rows,
)
from f1_ml_predictor.prediction.sprint import weekend_values as openf1_qualifying_values
from f1_ml_predictor.prediction.workspace import evaluate_contract, load_audited_history
from f1_ml_predictor.simulation import EventSimulation, simulate_championship
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.practice_capture import PRACTICE, captured_practice_values
from f1_ml_predictor.trust.prospective import load_bundle
from f1_ml_predictor.trust.qualifying_fallback import QUALIFYING_OPENF1
from f1_ml_predictor.trust.scheduler import scheduler_status
from f1_ml_predictor.trust.sprint_capture import (
    SPRINT_FIA,
    SPRINT_QUALIFYING,
    SPRINT_RESULT,
    capture_fia_positions,
    capture_grid,
    capture_points,
    capture_teams,
    captured_sprint_values,
    latest_capture,
)

STATUS = "development_only"
METHODOLOGY = "cutoff-specific-v3"
WARNING = (
    "DEVELOPMENT ONLY. Winner and podium models have not passed the frozen Gold selection "
    "gates, and nothing here has prospective confirmation. These are not validated race or "
    "championship forecasts."
)
FIXTURE = Path("docs/regression/c52b674-2026-round16-pre-qualifying.json")
_OUTPUT_ROOT = Path("data/predictions/development/next_race")
BOOTSTRAP_REPLICATES = 20
# Missing by the nature of the event (an unseen circuit), with flags well represented
# in training; these are never hidden even when every live driver lacks them.
STRUCTURAL_MISSING = ("driver_circuit_finish_mean", "circuit_dnf_rate", *SPRINT_NUMERIC)
SENSITIVITY_SIMULATIONS = 20000


def _write_json(path: Path, value: Any) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"unserializable {type(item).__name__}")

    data = json.dumps(value, sort_keys=True, indent=2, default=default, allow_nan=False)
    with path.open("xb") as handle:
        handle.write(data.encode("utf-8"))
    return file_sha256(path)


class ContractModel:
    """Live models for one contract, configured by its masked v3 evaluation."""

    def __init__(
        self,
        root: Path,
        history: AuditedHistory,
        datasets: dict[str, dict[str, Any]],
        contract: str,
        live_rows: list[dict[str, Any]],
        *,
        cutoff: datetime,
        seed: int,
        device: str,
        hardware: dict[str, Any] | None,
        progress: Callable[[str], None] | None,
        candidates: tuple[str, ...] = JOINT_CANDIDATES,
    ) -> None:
        self.contract = contract
        self.candidates = candidates
        dataset = datasets[contract]
        self.masked = tuple(
            name
            for name in numeric_features(contract)
            if name not in STRUCTURAL_MISSING
            and all(row[name] is None for row in live_rows)
            and any(row[name] is not None for row in dataset["rows"])
        )
        sha = dataset["manifest"]["dataset_sha256"]
        self.protocol, self.protocol_path = evaluate_contract(
            root,
            history,
            contract,
            dataset["rows"],
            sha,
            progress=progress,
            candidates=candidates,
        )
        if self.masked:
            self.evaluation, self.evaluation_path = evaluate_contract(
                root,
                history,
                contract,
                dataset["rows"],
                sha,
                masked=self.masked,
                progress=progress,
                candidates=candidates,
            )
        else:
            self.evaluation, self.evaluation_path = self.protocol, self.protocol_path
        self.dataset_sha256 = sha
        self.columns, self.dnf_columns = columns_for(contract, self.masked)
        self.training, self.training_events = training_rows(dataset["rows"], cutoff)
        for row in self.training:
            require_known_by(row["label_available_at"], cutoff)
        live = self.evaluation["live"]
        self.primary: str = live["primary"]
        self.members: list[str] = live["primary_members"]
        self.models: dict[str, StrengthModel] = {
            name: fit_strength(
                name, self.training, self.columns, seed=seed, device=device, hardware=hardware
            )
            for name in candidates
        }
        self.dnf_name: str = live["dnf_model"]
        self.dnf = fit_dnf(self.dnf_name, self.training, self.dnf_columns, seed)
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
        rng = np.random.default_rng(seed)
        order, retired = sample_mixture(components, dnf, draws=draws, rng=rng)
        values = marginals(order, retired)
        values["dnf_model"] = dnf
        if clean:
            # Same seed consumes the same random stream, so these are the same draws
            # with every retirement switched off: the order of a race nobody retires from.
            no_dnf = np.zeros_like(dnf)
            clean_order, clean_retired = sample_mixture(
                components, no_dnf, draws=draws, rng=np.random.default_rng(seed)
            )
            clean_values = marginals(clean_order, clean_retired)
            values["clean_finish"] = clean_values["finish"]
            values["clean_expected"] = clean_values["expected"]
        return values


def _race_table(
    rows: list[dict[str, Any]], values: dict[str, Any], draws: int
) -> list[dict[str, Any]]:
    table = []
    for index, row in enumerate(rows):
        finish = values["finish"][index]
        cumulative = np.cumsum(finish)
        table.append(
            {
                "driver_id": row["driver_id"],
                "constructor_id": row["constructor_id"],
                "winner_probability": float(values["winner"][index]),
                "podium_probability": float(values["podium"][index]),
                "dnf_probability": float(values["dnf"][index]),
                "dnf_model_probability": float(values["dnf_model"][index]),
                "finish_distribution": [float(value) for value in finish],
                "expected_position": float(values["expected"][index]),
                "most_likely_position": int(np.argmax(finish)) + 1,
                "position_interval_80": [
                    int(np.searchsorted(cumulative, quantile - 1e-12)) + 1
                    for quantile in (0.1, 0.9)
                ],
                "winner_draws": int(round(values["winner"][index] * draws)),
            }
        )
        if "clean_expected" in values:
            table[-1]["clean_expected_position"] = float(values["clean_expected"][index])
            table[-1]["clean_most_likely_position"] = (
                int(np.argmax(values["clean_finish"][index])) + 1
            )
    if "clean_expected" in values:
        # One P1..Pn order: clean-race expectation, ties broken by win probability.
        ranked = sorted(
            table, key=lambda item: (item["clean_expected_position"], -item["winner_probability"])
        )
        for position, item in enumerate(ranked, 1):
            item["predicted_position"] = position
    _check_race(table)
    return table


def _quality(live: dict[str, float], reference: list[dict[str, float]]) -> dict[str, Any]:
    """Compare live sharpness with the same model's historical out-of-fold forecasts."""
    result: dict[str, Any] = {"metrics": {}, "status": "pass", "notes": []}
    for key, value in live.items():
        history = np.asarray([item[key] for item in reference])
        percentile = float((history <= value).mean())
        result["metrics"][key] = {
            "live": value,
            "historical_percentile": percentile,
            "historical_min": float(history.min()),
            "historical_median": float(np.median(history)),
            "historical_max": float(history.max()),
        }
    sharper = [
        ("effective_win_contenders", "low"),
        ("winner_entropy", "low"),
        ("maximum_win_probability", "high"),
        ("fraction_below_0_1_percent_win", "high"),
    ]
    for key, side in sharper:
        item = result["metrics"][key]
        beyond = (
            item["live"] < item["historical_min"]
            if side == "low"
            else item["live"] > item["historical_max"]
        )
        extreme = (
            item["historical_percentile"] < 0.025
            if side == "low"
            else item["historical_percentile"] > 0.975
        )
        if beyond:
            result["status"] = "fail"
            result["notes"].append(f"{key} is sharper than every historical out-of-fold race")
        elif extreme and result["status"] == "pass":
            result["status"] = "warn"
            result["notes"].append(f"{key} is in the sharpest 2.5% of historical races")
    flat = result["metrics"]["effective_win_contenders"]["historical_percentile"]
    if flat > 0.975:
        result["notes"].append(
            "informational: flatter than 97.5% of historical out-of-fold races, consistent "
            "with hidden predictors and an unseen circuit"
        )
    return result


def _bootstrap_spread(
    model: ContractModel, name: str, rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    """Race-level bootstrap refits: parameter uncertainty in calibrated utility units."""
    events = sorted({row["event_id"] for row in model.training})
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in model.training:
        grouped.setdefault(row["event_id"], []).append(row)
    rng = np.random.default_rng((seed, 99))
    tau, _, _ = model.calibration(name, False)
    utilities = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sample = rng.choice(len(events), size=len(events), replace=True)
        rows_sample = [
            {**row, "event_id": f"{row['event_id']}#{replicate}-{draw}"}
            for draw, index in enumerate(sample)
            for row in grouped[events[index]]
        ]
        refit = fit_strength(name, rows_sample, model.columns, seed=seed)
        utility = refit.utility(rows) / tau
        utilities.append(utility - utility.mean())
    spread = np.std(np.asarray(utilities), axis=0)
    return {
        "model": name,
        "replicates": BOOTSTRAP_REPLICATES,
        "mean_utility_standard_deviation": float(spread.mean()),
        "max_utility_standard_deviation": float(spread.max()),
        "by_driver": {
            row["driver_id"]: float(value) for row, value in zip(rows, spread, strict=True)
        },
    }


def _feature_drivers(model: ContractModel, experimental: str) -> dict[str, Any]:
    """Global drivers: standardized logistic coefficients and booster split importance."""
    result: dict[str, Any] = {}
    if "logistic_pl" in model.models:
        logistic = model.models["logistic_pl"].model.named_steps["model"]
        coefficients = sorted(
            zip(model.columns, logistic.coef_[0], strict=True), key=lambda item: -abs(item[1])
        )
        result["logistic_pl_standardized_coefficients"] = [
            {"feature": name, "coefficient": float(value)} for name, value in coefficients[:10]
        ]
    fitted = model.models[experimental].model
    booster = fitted[1] if isinstance(fitted, tuple) else None
    values = getattr(booster, "feature_importances_", None)
    if values is not None and float(np.sum(values)) > 0:
        share = np.asarray(values, dtype=float) / float(np.sum(values))
        ranked = sorted(zip(model.columns, share, strict=True), key=lambda item: -item[1])
        result[f"{experimental}_importance_share"] = [
            {"feature": name, "share": float(value)} for name, value in ranked[:10]
        ]
    return result


def _season(
    history: AuditedHistory,
    model: ContractModel,
    next_model: ContractModel,
    next_rows: list[dict[str, Any]],
    schedule: list[Any],
    target: Any,
    cutoff: datetime,
    roster: dict[str, str],
    *,
    worlds: int,
    orders_per_world: int,
    sigma: float,
    drift_variance: float,
    temperature_scale: float,
    members: list[str] | None,
    seed: int,
    run_id: str,
    source_hash: str,
    next_kind: str = "race",
    sprint_counted: bool = False,
) -> tuple[list[EventSimulation], list[dict[str, Any]]]:
    """Event-specific pre-weekend distributions sampled within shared strength worlds.

    Each world draws persistent driver offsets and a random-walk drift path; the
    next race has lag zero, later weekends accumulate per-race drift variance. The
    published session (``next_kind``, the race or the sprint) uses this run's rows.
    """
    remaining = remaining_sessions(schedule, target, cutoff, sprint_counted=sprint_counted)
    weekends_ahead = sorted({item.event for item, _, _ in remaining}, key=lambda event: event.round)
    lag = {event: index for index, event in enumerate(weekends_ahead)}
    rng = np.random.default_rng((seed, 7))
    count = len(roster)
    persistent = rng.normal(0.0, sigma, size=(worlds, 1, count)) if sigma > 0 else 0.0
    steps = rng.normal(0.0, np.sqrt(drift_variance), size=(worlds, len(weekends_ahead), count))
    steps[:, 0, :] = 0.0
    paths = np.cumsum(steps, axis=1) + persistent
    shared = sigma > 0 or drift_variance > 0
    events, sessions = [], []
    for item, kind, start in remaining:
        is_next = item.event == target.event and kind == next_kind
        if is_next:
            rows, owner = next_rows, next_model
        else:
            weekend = history.weekends[item.event]
            rows, _ = build_rows(
                history,
                "pre_weekend",
                event=item.event,
                circuit_id=weekend.circuit_id,
                cutoff=cutoff,
                roster=roster,
            )
            owner = model
        unseen = not bool(rows[0]["circuit_seen_before"])
        components = owner.components(
            rows, unseen, members=members, temperature_scale=temperature_scale
        )
        dnf = owner.dnf.predict(rows)
        draws = worlds * orders_per_world
        session_rng = np.random.default_rng((seed, item.event.round, kind == "sprint"))
        expanded = (
            np.repeat(paths[:, lag[item.event], :], orders_per_world, axis=0) if shared else None
        )
        order, _ = sample_mixture(components, dnf, draws=draws, rng=session_rng, offsets=expanded)
        drivers = [row["driver_id"] for row in rows]
        orders = tuple(tuple(drivers[index] for index in sampled) for sampled in order)
        rules, rules_status = points_rules(history.ledger, item.event, kind)
        model_id = f"{run_id}:{owner.contract}:{METHODOLOGY}"
        events.append(
            EventSimulation(
                event_id=item.event,
                scheduled_at=start,
                available_at=cutoff,
                driver_constructors=roster,
                sampled_orders=orders,
                points_eligible_samples=tuple(tuple(roster) for _ in orders),
                rules=rules,
                source_hash=source_hash,
                model_id=model_id,
                eligibility_policy=ELIGIBILITY_POLICY,
                sample_groups=tuple(np.repeat(np.arange(worlds), orders_per_world).tolist()),
            )
        )
        sessions.append(
            {
                "event_id": item.event.partition(),
                "race_name": item.race_name,
                "session": kind,
                "scheduled_at": start.isoformat(),
                "circuit_id": rows[0]["circuit_id"],
                "circuit_seen_before": not unseen,
                "contract": owner.contract,
                "model_id": model_id,
                "rules_id": rules.rules_id,
                "rules_status": rules_status,
                "points_by_position": list(rules.points_by_position),
                "orders": draws,
                "drift_lag": lag[item.event],
            }
        )
    return events, sessions


def _captured_sprint_roster(
    root: Path,
    rows: list[dict[str, Any]],
    event: EventId,
    grid_record: dict[str, Any],
    cutoff: datetime,
    label: str = "sprint_qualifying",
) -> tuple[dict[str, str], str]:
    """Drivers from an OpenF1 session capture with their constructors that weekend.

    The capture's OpenF1 session team decides; the latest audited roster, then the
    latest Jolpica entry this season, cover a team name the table does not map.
    """
    drivers = capture_grid(grid_record).positions
    session_teams = capture_teams(grid_record)
    latest, roster_basis = _latest_roster(rows, event.season, cutoff)
    unknown = sorted(set(drivers) - set(session_teams) - set(latest))
    fallback = latest_jolpica_constructors(root, event, unknown)
    if set(unknown) - set(fallback):
        missing = sorted(set(unknown) - set(fallback))
        raise ValueError(f"sprint qualifying drivers have no known constructor: {missing}")
    roster = {
        driver: session_teams.get(driver) or latest.get(driver) or fallback[driver]
        for driver in drivers
    }
    teams = list(roster.values())
    crowded = sorted(team for team in set(teams) if teams.count(team) > 2)
    if crowded:
        raise ValueError(f"sprint roster gives more than two drivers to {crowded}")
    moved = sorted(
        driver for driver, team in session_teams.items() if latest.get(driver, team) != team
    )
    source = (
        f"captured_{label} drivers; constructors from the captured OpenF1 session "
        f"teams, else {roster_basis}"
        + (f"; seat changes since {roster_basis}: {', '.join(moved)}" if moved else "")
        + (f"; latest Jolpica entry for {', '.join(unknown)}" if unknown else "")
    )
    return roster, source


def _canonical_capture_base(
    root: Path, capture_record: dict[str, Any], base: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Certified capture rows keyed by canonical Gold driver IDs.

    The capture keeps Jolpica driver IDs; Gold names drivers through the FIA name
    table (Jolpica ``arvid_lindblad`` is Gold ``lindblad``). The capture's own
    Jolpica qualifying payload supplies the names, as the sprint crosswalk does.
    """
    manifest, tables = load_bundle(root / capture_record["bundle"])
    drivers: list[dict[str, Any]] = []
    for name, request in manifest["request_metadata"]["source_requests"].items():
        if isinstance(request, dict) and request.get("role") == "qualifying":
            payload = json.loads(tables[name]["payload_json"][0].as_py())
            for race in payload["MRData"]["RaceTable"]["Races"]:
                drivers.extend(row["Driver"] for row in race["QualifyingResults"])
    identities = crosswalk({"MRData": {"DriverTable": {"Drivers": drivers}}})
    mapped = {
        identities.driver(driver): {**row, "driver_id": identities.driver(driver)}
        for driver, row in base.items()
    }
    if len(mapped) != len(base):
        raise ValueError("certified capture drivers collide after canonical mapping")
    return mapped


def predict_next_race(
    root: Path,
    *,
    season: int | None = None,
    simulations: int = 100000,
    seed: int = 42,
    draws: int = 65536,
    worlds: int = 1000,
    orders_per_world: int = 16,
    device: str = "auto",
    gold_dir: Path | None = None,
    dnf_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
    progress: Callable[[str], None] | None = None,
    candidates: tuple[str, ...] = JOINT_CANDIDATES,
    session: str = "auto",
) -> dict[str, Any]:
    """Freeze a cutoff-specific snapshot and publish development race and title outputs.

    ``session`` chooses what is published: ``race``, ``sprint`` (after a captured
    sprint qualifying and before the sprint) or ``auto``, which publishes the sprint
    in that window and the race otherwise.
    """
    if session not in {"auto", "race", "sprint"}:
        raise ValueError("session must be auto, race or sprint")
    root = root.resolve()
    clock = (now or (lambda: datetime.now(UTC)))()
    require_utc(clock, "prediction clock")
    state = scheduler_status(root)
    observation = state.get("schedule_observation")
    if not observation:
        raise ValueError("no retained schedule observation; run collect-next-race first")
    schedule, schedule_source = load_schedule(root, observation)
    payload = json.loads((root / observation).read_text(encoding="utf-8"))["payload"]
    observed_at = datetime.fromisoformat(schedule_source["captured_at"])
    require_known_by(observed_at, clock)
    upcoming = [
        item
        for item in schedule
        if item.race_start > clock and (season is None or item.event.season == season)
    ]
    if not upcoming:
        raise ValueError("the retained schedule has no upcoming race")
    target = upcoming[0]
    history, dnf_version, sources = load_audited_history(root, gold_dir=gold_dir, dnf_dir=dnf_dir)
    live_weekends = weekends(payload)
    history.weekends.update(live_weekends)
    for version in (history.version, dnf_version):
        if version.directory.name != version.dataset_version:
            raise ValueError("training data must come from an immutable dataset version")
    weekend = live_weekends[target.event]
    if weekend.circuit_id != target.circuit_id:
        raise ValueError("schedule circuit identities disagree")
    capture = _capture_base(root, state, target, clock)
    notes = []
    sprint_grid_capture = None
    if weekend.sprint is not None and clock < weekend.sprint and session != "race":
        sprint_grid_capture = latest_capture(root, target.event, SPRINT_QUALIFYING, clock)
    if session == "sprint" and sprint_grid_capture is None:
        raise ValueError(
            "no captured sprint qualifying before the sprint; run collect-next-race after "
            "sprint qualifying"
        )
    is_sprint = sprint_grid_capture is not None
    qualifying_fallback = (
        latest_capture(root, target.event, QUALIFYING_OPENF1, clock)
        if capture is None and not is_sprint and session != "sprint"
        else None
    )
    sprint_grid = None
    if sprint_grid_capture is not None:
        assert weekend.sprint is not None
        contract = SPRINT_CONTRACT
        sprint_grid = capture_grid(sprint_grid_capture)
        cutoff = sprint_grid.available_at
        capture_record = None
        roster, roster_source = _captured_sprint_roster(
            root, history.rows, target.event, sprint_grid_capture, cutoff
        )
        weekend_values = None
    elif qualifying_fallback is not None:
        # Jolpica had not published qualifying: a provisional post-qualifying run from
        # the frozen OpenF1 qualifying result, never a certified capture.
        contract, capture_record = "post_qualifying", None
        qualifying_grid = capture_grid(qualifying_fallback)
        cutoff = qualifying_grid.available_at
        roster, roster_source = _captured_sprint_roster(
            root, history.rows, target.event, qualifying_fallback, cutoff, "openf1_qualifying"
        )
        weekend_values = openf1_qualifying_values(qualifying_grid, roster)
        notes.append(
            "PROVISIONAL: qualifying comes from the OpenF1 session result frozen at "
            f"{cutoff.isoformat()} because Jolpica had not published it; not certified and "
            "not evaluation-eligible"
        )
    elif capture is None:
        contract, cutoff, capture_record = "pre_weekend", clock, None
        grid_record = (
            latest_capture(root, target.event, SPRINT_QUALIFYING, cutoff)
            if weekend.sprint is not None
            else None
        )
        if grid_record is not None:
            # This weekend's sprint qualifying names who is racing, so it beats the
            # latest audited roster when a seat changed.
            roster, roster_source = _captured_sprint_roster(
                root, history.rows, target.event, grid_record, cutoff
            )
        else:
            roster, roster_basis = _latest_roster(history.rows, target.event.season, cutoff)
            roster_source = f"latest_audited_gold_event_roster:{roster_basis}"
        weekend_values = None
        if weekend.first_practice is not None and clock >= weekend.first_practice:
            notes.append(
                "weekend sessions before the cutoff are not captured, so the pre-weekend "
                "contract is used"
            )
    else:
        capture_record, base = capture
        base = _canonical_capture_base(root, capture_record, base)
        contract = "pre_race" if capture_record["cutoff_kind"] == "pre_race" else "post_qualifying"
        cutoff = next(iter(base.values()))["prediction_timestamp"]
        roster = {driver: row["constructor_id"] for driver, row in base.items()}
        roster_source = "certified_capture"
        allowed = (*QUALIFYING_NUMERIC, *(GRID_NUMERIC if contract == "pre_race" else ()))
        weekend_values = {
            driver: {
                **{name: row.get(name) for name in allowed},
                **{f"{name}_available_at": row["feature_timestamp"] for name in allowed},
            }
            for driver, row in base.items()
        }
    if weekend_values is not None and weekend.sprint is not None and weekend.sprint <= cutoff:
        live_sprint = captured_sprint_values(
            latest_capture(root, target.event, SPRINT_QUALIFYING, cutoff),
            latest_capture(root, target.event, SPRINT_RESULT, cutoff),
        )
        for driver, values in weekend_values.items():
            values.update(live_sprint.get(driver, {}))
    require_known_by(observed_at, cutoff)
    live_practice: dict[str, dict[str, Any]] = {}
    practice_source = None
    if contract in {SPRINT_CONTRACT, "post_qualifying", "pre_race"}:
        live_practice, practice_source = captured_practice_values(
            root, latest_capture(root, target.event, PRACTICE, clock), roster, cutoff
        )
        if practice_source is not None:
            notes.append(
                f"practice predictors use FIA practice {practice_source['session']} "
                f"classification document {practice_source['document_id']}, published by "
                f"{practice_source['available_at']}"
            )
            if weekend_values is not None:
                for driver, values in weekend_values.items():
                    values.update(live_practice.get(driver, {}))
    current_sprint = None
    if not is_sprint and weekend.sprint is not None and weekend.sprint <= cutoff:
        result = latest_capture(root, target.event, SPRINT_RESULT, clock)
        if result is None:
            raise ValueError(
                "this weekend's sprint has run but its result is not captured; run "
                "collect-next-race"
            )
        current_sprint = {
            "points": capture_points(result),
            "captured_at": result["captured_at"],
            "bundle": result["bundle"],
            "provider": result.get("provider", "jolpica"),
        }
        fia = latest_capture(root, target.event, SPRINT_FIA, cutoff)
        if fia is not None:
            current_sprint["fia"] = {
                "positions": capture_fia_positions(root, fia),
                "captured_at": fia["captured_at"],
                "bundle": fia["bundle"],
                "document_id": fia["document"]["document_id"],
            }
    hardware_path = root / "models/experiments/hardware.json"
    hardware = (
        json.loads(hardware_path.read_text(encoding="utf-8"))
        if device != "cpu" and hardware_path.exists()
        else None
    )
    datasets = build_contract_datasets(root, history, dnf_version, sources)
    sprint_details: dict[str, Any] | None = None
    if is_sprint:
        assert sprint_grid is not None and weekend.sprint is not None
        gold = gold_sprints(root, history, clock=cutoff)
        if gold is not None:
            sprints, excluded_sprints, sprint_source = gold
            sprint_tier = "Gold"
            notes.append(
                "sprint calibration uses the FIA-audited Gold sprint history; strength models "
                "are trained on Gold races"
            )
        else:
            sprints, excluded_sprints = historical_sprints(root, history, clock=cutoff)
            sprint_source = {"kind": "development_jolpica_openf1"}
            sprint_tier = "Development"
            notes.append(
                "sprint calibration and sprint history are Development tier (Jolpica sprint "
                "results and OpenF1 sprint qualifying, cross-checked against audited FIA sprint "
                "points); strength models are trained on Gold races"
            )
        rows, reasons = sprint_rows(
            history,
            SprintEvent(
                target.event,
                target.circuit_id,
                cutoff,
                weekend.sprint,
                roster,
                sprint_grid,
                {},
                live_practice,
            ),
            labelled=False,
        )
        sprint_details = {
            "sprints": sprints,
            "excluded": excluded_sprints,
            "tier": sprint_tier,
            "source": sprint_source,
        }
    else:
        rows, reasons = build_rows(
            history,
            contract,
            event=target.event,
            circuit_id=target.circuit_id,
            cutoff=cutoff,
            roster=roster,
            weekend=weekend_values,
        )
    appeals = {
        str(item["points_status"]).removeprefix("published_pending_appeal:")
        for item in reasons.values()
        if str(item.get("points_status")).startswith("published_pending_appeal:")
    }
    for events_under_appeal in sorted(appeals):
        notes.append(
            "championship point features use published points still under appeal for "
            + events_under_appeal
        )
    common: dict[str, Any] = {
        "cutoff": cutoff,
        "seed": seed,
        "device": device,
        "hardware": hardware,
        "candidates": candidates,
    }
    model: Any
    if is_sprint:
        assert sprint_details is not None
        race_dataset = datasets["post_qualifying"]
        race_sha = race_dataset["manifest"]["dataset_sha256"]
        sprint_evaluation, sprint_path = evaluate_sprints(
            root,
            history,
            race_dataset["rows"],
            race_sha,
            sprint_details["sprints"],
            sprint_details["excluded"],
            seed=seed,
            candidates=candidates,
            progress=progress,
            tier=sprint_details["tier"],
        )
        model = SprintModel(
            sprint_evaluation,
            race_dataset["rows"],
            sprint_details["sprints"],
            cutoff=cutoff,
            seed=seed,
            candidates=candidates,
            evaluation_path=sprint_path,
            dataset_sha256=race_sha,
        )
    else:
        model = ContractModel(root, history, datasets, contract, rows, progress=progress, **common)
    if contract == "pre_weekend":
        pre_model, pre_rows = model, rows
    else:
        pre_rows, _ = build_rows(
            history,
            "pre_weekend",
            event=target.event,
            circuit_id=target.circuit_id,
            cutoff=cutoff,
            roster=roster,
        )
        pre_model = ContractModel(
            root, history, datasets, "pre_weekend", pre_rows, progress=progress, **common
        )
    unseen = not bool(rows[0]["circuit_seen_before"])
    final = model.distribution(
        rows, model.components(rows, unseen), draws=draws, seed=seed, clean=True
    )
    race = _race_table(rows, final, draws)
    stages = {
        "uncalibrated": model.distribution(
            rows, model.components(rows, unseen, calibrated=False), draws=draws, seed=seed
        ),
        "calibrated_shared": model.distribution(
            rows,
            model.components(rows, unseen, allow_unseen_rule=False),
            draws=draws,
            seed=seed,
        ),
    }
    objectives = model.evaluation["live"]["mean_objective"]
    boosted = [name for name in BACKENDS if name in candidates]
    experimental = min(boosted or list(candidates), key=lambda name: objectives[name])
    alternatives = {
        name: model.distribution(
            rows, model.components(rows, unseen, members=[name]), draws=draws, seed=seed
        )
        for name in candidates
    }
    baselines = baseline_marginals(model.training, rows, model.columns, seed)
    live_sharpness = sharpness(final["winner"], final["podium"])
    quality = _quality(live_sharpness, model.evaluation["live"]["sharpness_reference"])
    ood = ood_report(
        model.training,
        rows,
        model.columns,
        circuit_id=target.circuit_id,
        circuit_name=weekend.circuit_name,
    )
    run_id = uuid.uuid4().hex
    run_dir = root / _OUTPUT_ROOT / target.event.partition() / contract / run_id
    run_dir.mkdir(parents=True)
    model_path = root / "models/development/next_race" / run_id / "models.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "contract": {"models": model.models, "dnf": model.dnf},
            "pre_weekend": {"models": pre_model.models, "dnf": pre_model.dnf},
        },
        model_path,
    )
    model_sha = file_sha256(model_path)
    pre_live = pre_model.evaluation["live"]
    persistence = pre_live["persistent_strength"]
    sigma = float(persistence["persistent_standard_deviation"])
    strength = pre_live["strength_model_for_uncertainty"]
    simulated_tau = pre_live["simulation_temperature"].get("temperature")
    scale = (
        float(simulated_tau) / float(pre_live["calibration"][strength]["temperature"])
        if sigma > 0 and simulated_tau
        else 1.0
    )
    drift = estimate_form_drift(
        root,
        history,
        datasets["pre_weekend"]["rows"],
        pre_model.evaluation,
        masked=pre_model.masked,
        seed=seed,
        dataset_sha256=pre_model.dataset_sha256,
    )
    drift_variance = float(drift["per_race_variance"])
    standings, standings_notes, standings_sources = published_standings(
        history.ledger, target.event, cutoff, roster, current_sprint=current_sprint
    )
    season_args: dict[str, Any] = {
        "worlds": worlds,
        "orders_per_world": orders_per_world,
        "seed": seed,
        "run_id": run_id,
        "source_hash": model_sha,
        "next_kind": "sprint" if is_sprint else "race",
        "sprint_counted": current_sprint is not None,
    }
    events, sessions = _season(
        history,
        pre_model,
        model,
        rows,
        schedule,
        target,
        cutoff,
        roster,
        sigma=sigma,
        drift_variance=drift_variance,
        temperature_scale=scale,
        members=None,
        **season_args,
    )
    championship = simulate_championship(
        standings, events, prediction_timestamp=cutoff, simulations=simulations, seed=seed
    )
    sensitivity: dict[str, Any] = {}
    fixed_events, _ = _season(
        history,
        pre_model,
        model,
        rows,
        schedule,
        target,
        cutoff,
        roster,
        sigma=0.0,
        drift_variance=0.0,
        temperature_scale=1.0,
        members=None,
        **season_args,
    )
    fixed = simulate_championship(
        standings,
        fixed_events,
        prediction_timestamp=cutoff,
        simulations=SENSITIVITY_SIMULATIONS,
        seed=seed,
    )
    sensitivity["fixed_strength_no_persistent_uncertainty"] = {
        "wdc": dict(fixed.wdc.title_probability),
        "wcc": dict(fixed.wcc.title_probability),
    }
    ranked = sorted(candidates, key=lambda name: pre_live["mean_objective"][name])
    for name in ranked[:3]:
        alt_events, _ = _season(
            history,
            pre_model,
            model,
            rows,
            schedule,
            target,
            cutoff,
            roster,
            sigma=sigma,
            drift_variance=drift_variance,
            temperature_scale=scale,
            members=[name],
            **season_args,
        )
        alt = simulate_championship(
            standings,
            alt_events,
            prediction_timestamp=cutoff,
            simulations=SENSITIVITY_SIMULATIONS,
            seed=seed,
        )
        sensitivity[f"single_candidate_{name}"] = {
            "wdc": dict(alt.wdc.title_probability),
            "wcc": dict(alt.wcc.title_probability),
        }
    bootstrap = _bootstrap_spread(pre_model, strength, pre_rows, seed)
    created_at = datetime.now(UTC)
    identity = {
        "validation_status": STATUS,
        "methodology": METHODOLOGY,
        "model_run_id": run_id,
        "cutoff_kind": contract,
        "prediction_timestamp_utc": cutoff.isoformat(),
        "dataset_version": history.version.dataset_version,
        "dnf_dataset_version": dnf_version.dataset_version,
        "feature_contract": CONTRACT_VERSION,
        "evaluation_protocol": PROTOCOL_V3_VERSION,
        "session": "sprint" if is_sprint else "race",
    }
    by_driver = {row["driver_id"]: index for index, row in enumerate(rows)}
    table_rows = []
    for item in race:
        index = by_driver[item["driver_id"]]
        table_rows.append(
            {
                **identity,
                "event_id": target.event.partition(),
                "race_name": target.race_name,
                "prediction_timestamp": cutoff,
                "generated_at": created_at,
                **{key: item[key] for key in item if key != "position_interval_80"},
                "primary_model": model.primary,
                "logistic_win_probability": float(baselines["logistic"]["winner"][index]),
                "logistic_podium_probability": float(baselines["logistic"]["podium"][index]),
                "linear_finish_rank": float(baselines["logistic"]["expected"][index]),
                "heuristic_win_probability": float(baselines["heuristic"]["winner"][index]),
                "experimental_model": experimental,
                "experimental_win_probability": float(alternatives[experimental]["winner"][index]),
                "uncalibrated_win_probability": float(stages["uncalibrated"]["winner"][index]),
            }
        )
    predictions_path = run_dir / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(table_rows), predictions_path)
    race_payload = {
        **identity,
        "event_id": target.event.partition(),
        "race_name": target.race_name,
        "race_start": target.race_start.isoformat(),
        "session_start": (
            weekend.sprint if is_sprint and weekend.sprint is not None else target.race_start
        ).isoformat(),
        "draws": draws,
        "seed": seed,
        "monte_carlo_resolution": 1 / draws,
        "simulation_standard_error_max": 0.5 / draws**0.5,
        "primary_model": model.primary,
        "primary_members": model.members,
        "calibration": {
            name: dict(
                zip(
                    ("temperature", "shrink", "source"),
                    model.calibration(name, unseen),
                    strict=True,
                )
            )
            for name in model.members
        },
        "drivers": race,
        "stages": {
            name: {
                row["driver_id"]: {
                    "win": float(values["winner"][index]),
                    "podium": float(values["podium"][index]),
                    "expected_finish": float(values["expected"][index]),
                }
                for index, row in enumerate(rows)
            }
            for name, values in {
                **stages,
                **{f"candidate_{k}": v for k, v in alternatives.items()},
            }.items()
        },
        "baselines": {
            name: {
                row["driver_id"]: {key: float(values[key][index]) for key in values}
                for index, row in enumerate(rows)
            }
            for name, values in baselines.items()
        },
    }
    race_sha = _write_json(run_dir / "race_distribution.json", race_payload)
    championship_payload = {
        **identity,
        "validated_forecast": False,
        "race_distribution_sha256": race_sha,
        "starting_points": standings.to_dict(),
        "standings_source": {
            "scoring_ledger_sha256": history.ledger.sha256,
            "rounds": standings_sources,
            "notes": standings_notes,
        },
        "sessions": sessions,
        "uncertainty": {
            "worlds": worlds,
            "orders_per_world": orders_per_world,
            "persistent_strength": persistence,
            "persistent_standard_deviation": sigma,
            "form_drift": drift,
            "per_race_drift_variance": drift_variance,
            "simulation_temperature_scale": scale,
            "bootstrap_parameter_spread": bootstrap,
            "monte_carlo": "simulations resample whole worlds; title-probability Monte Carlo "
            "error is bounded by treating the worlds as the effective sample, "
            "sqrt(p(1-p)/worlds), not by the simulation count",
            "monte_carlo_standard_error_max": 0.5 / worlds**0.5,
            "sensitivity": sensitivity,
        },
        "assumptions": [
            "Every remaining event uses its own pre-weekend rows: circuit history, circuit "
            "attrition and sprint format change by event; driver form is the form known at "
            "the cutoff.",
            "Each simulated world draws a driver strength random walk across the remaining "
            "weekends, with per-race variance validated on stale-form historical forecasts, "
            "plus any persistent offset supported by out-of-fold residual correlation.",
            "Simulated results do not update rolling form features; drift between events is "
            "carried by the shared random walk instead.",
            "Sprints reuse the race model with the audited sprint points table.",
            "Full-distance tables of the latest audited season rule; shortened races, "
            "cancellations and penalties are not simulated.",
            "Every entered driver is points eligible at their sampled position.",
            "The current roster races every remaining round.",
        ],
        "simulator": championship.to_dict(),
    }
    championship_sha = _write_json(run_dir / "championship.json", championship_payload)
    comparison = _compare(root, target, race, rows, stages, championship_payload)
    if comparison is not None:
        _write_json(run_dir / "comparison.json", {**identity, **comparison})
    checks = _validate(
        root,
        race,
        rows,
        cutoff,
        model,
        pre_model,
        events,
        model_sha,
        run_id,
        [predictions_path, run_dir / "race_distribution.json", run_dir / "championship.json"],
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    evaluation = model.evaluation
    manifest = {
        **identity,
        "warning": WARNING,
        "validated_forecast": False,
        "created_at": created_at.isoformat(),
        "git_commit": commit,
        "notes": notes,
        "event": {
            "event_id": target.event.partition(),
            "race_name": target.race_name,
            "circuit_id": target.circuit_id,
            "circuit_name": weekend.circuit_name,
            "race_start": target.race_start.isoformat(),
            "first_practice": None
            if weekend.first_practice is None
            else weekend.first_practice.isoformat(),
            "qualifying_start": None
            if weekend.qualifying is None
            else weekend.qualifying.isoformat(),
            "sprint_weekend": weekend.sprint is not None,
            "sprint_qualifying_start": None
            if weekend.sprint_qualifying is None
            else weekend.sprint_qualifying.isoformat(),
            "sprint_start": None if weekend.sprint is None else weekend.sprint.isoformat(),
        },
        "schedule_source": schedule_source,
        "roster_source": roster_source,
        "capture": None
        if capture_record is None
        else {key: capture_record[key] for key in ("bundle", "manifest_sha256", "captured_at")},
        "qualifying_fallback": None
        if qualifying_fallback is None
        else {
            "provider": "openf1",
            "bundle": qualifying_fallback["bundle"],
            "captured_at": qualifying_fallback["captured_at"],
            "certified": False,
        },
        "evidence_tier": {
            "training": "Gold",
            "prediction_snapshot": "Development",
            **(
                {"sprint_calibration": sprint_details["tier"]}
                if is_sprint and sprint_details is not None
                else {}
            ),
        },
        "sprint": None
        if sprint_details is None or sprint_grid_capture is None
        else {
            "addendum": addendum_for(sprint_details["tier"])[0]["version"],
            "addendum_sha256": addendum_for(sprint_details["tier"])[1],
            "history_tier": sprint_details["tier"],
            "history_source": sprint_details["source"],
            "grid_capture": sprint_grid_capture["bundle"],
            "grid_source": None if sprint_grid is None else sprint_grid.source,
            "historical_sprints": [item.event.partition() for item in sprint_details["sprints"]],
            "excluded_sprints": sprint_details["excluded"],
            "sprint_labels_known": model.sprint_labels,
        },
        "current_sprint_points": None
        if current_sprint is None
        else {
            "bundle": current_sprint["bundle"],
            "captured_at": current_sprint["captured_at"],
            "provider": current_sprint["provider"],
            "fia": None
            if "fia" not in current_sprint
            else {
                key: current_sprint["fia"][key] for key in ("bundle", "captured_at", "document_id")
            },
        },
        "dataset": {
            "gold": history.version.dataset_version,
            "gold_manifest_sha256": history.version.manifest_sha256,
            "dnf": dnf_version.dataset_version,
            "dnf_manifest_sha256": dnf_version.manifest_sha256,
            "dnf_known_labels": sum(row["label_dnf"] is not None for row in dnf_version.rows),
            "contract_dataset_sha256": model.dataset_sha256,
            "scoring_ledger_sha256": history.ledger.sha256,
            "schedule_sources": sources,
        },
        "protocol": {"version": PROTOCOL_V3_VERSION, "sha256": PROTOCOL_V3_SHA256},
        "evaluation": {
            "protocol_evaluation": model.protocol_path.relative_to(root).as_posix(),
            "live_masked_evaluation": model.evaluation_path.relative_to(root).as_posix(),
            "masked_for_training": list(model.masked),
            "outer_races": evaluation["outer_races"],
            "primary": model.primary,
            "primary_members": model.members,
            "experimental_model": experimental,
            "mean_objective": evaluation["live"]["mean_objective"],
            "formal_gate": {
                name: {task: value["status"] for task, value in result["formal_gate"].items()}
                for name, result in evaluation["models"].items()
            },
            "protocol_formal_gate": {
                name: {task: value["status"] for task, value in result["formal_gate"].items()}
                for name, result in model.protocol["models"].items()
            },
            "dnf": {
                "model": model.dnf_name,
                "candidates": {
                    name: {
                        key: value[key]
                        for key in (
                            "brier_score",
                            "log_loss",
                            "ece",
                            "labels",
                            "mean_within_race_standard_deviation",
                        )
                    }
                    for name, value in evaluation["dnf"]["candidates"].items()
                },
            },
        },
        "model_artifact": {"path": model_path.relative_to(root).as_posix(), "sha256": model_sha},
        "missing_feature_reasons": {
            row["driver_id"]: reasons[row["driver_id"]]["missing_reasons"] for row in rows
        },
        "ood": ood,
        "feature_drivers": _feature_drivers(model, experimental),
        "sharpness": {"live": live_sharpness, "quality": quality},
        "seeds": {"race": seed, "championship": seed},
        "race_draws": draws,
        "championship": {"simulations": simulations, "sha256": championship_sha},
        "execution": {
            "requested_device": device,
            "devices": sorted({item.device for item in model.models.values()}),
            "device_reasons": sorted({item.device_reason for item in model.models.values()}),
            "evaluation_devices": evaluation.get("devices"),
            "libraries": library_versions(),
        },
        "artifacts": {
            "predictions": {
                "path": predictions_path.relative_to(root).as_posix(),
                "sha256": file_sha256(predictions_path),
            },
            "race_distribution": {"sha256": race_sha},
            "championship": {"sha256": championship_sha},
        },
        "validation_checks": checks,
        "development_quality": quality["status"],
    }
    manifest_sha = _write_json(run_dir / "manifest.json", manifest)
    report = render_report(
        manifest,
        race_payload,
        championship_payload,
        model.evaluation,
        model.protocol,
        comparison,
        rows,
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "status": STATUS,
        "run_dir": run_dir.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha,
        "event_id": target.event.partition(),
        "cutoff_kind": contract,
        "prediction_timestamp_utc": cutoff.isoformat(),
        "report": report,
    }


def _compare(
    root: Path,
    target: Any,
    race: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    stages: dict[str, dict[str, Any]],
    championship: dict[str, Any],
) -> dict[str, Any] | None:
    """Attribute movements from the c52b674 forecast to model, calibration and OOD stages."""
    path = root / FIXTURE
    if not path.exists():
        return None
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture["event_id"] != target.event.partition():
        return None
    index = {row["driver_id"]: position for position, row in enumerate(rows)}
    drivers = {}
    for item in race:
        driver = item["driver_id"]
        old = fixture["drivers"].get(driver)
        if old is None:
            continue
        slot = index[driver]
        uncal = float(stages["uncalibrated"]["winner"][slot])
        shared = float(stages["calibrated_shared"]["winner"][slot])
        drivers[driver] = {
            "old": old,
            "new": {
                "win": item["winner_probability"],
                "podium": item["podium_probability"],
                "dnf": item["dnf_model_probability"],
                "expected_finish": item["expected_position"],
            },
            "win_stages": {
                "c52b674": old["win"],
                "new_models_and_features_uncalibrated": uncal,
                "after_calibration": shared,
                "final": item["winner_probability"],
            },
            "win_attribution": {
                "models_and_features": uncal - old["win"],
                "calibration": shared - uncal,
                "unseen_circuit_rule": item["winner_probability"] - shared,
            },
            "features": {
                key: rows[slot].get(key)
                for key in (
                    "recent_finish_mean_3",
                    "driver_finish_mean_any_10",
                    "driver_qualifying_mean_any_5",
                    "constructor_average_finish_last_5",
                    "driver_dnf_rate_any_10",
                    "constructor_dnf_rate_any_10",
                )
            },
        }
    simulator = championship["simulator"]
    return {
        "fixture": FIXTURE.as_posix(),
        "fixture_run_id": fixture["model_run_id"],
        "old_temperature": fixture["temperature"],
        "old_position_backend": fixture["position_backend"],
        "drivers": drivers,
        "wdc": {
            name: {
                "old": fixture["wdc"].get(name, {}).get("title"),
                "new": simulator["wdc"]["title_probability"][name],
                "old_expected_points": fixture["wdc"].get(name, {}).get("expected_points"),
                "new_expected_points": simulator["wdc"]["mean_final_points"][name],
            }
            for name in simulator["wdc"]["title_probability"]
        },
        "wcc": {
            name: {
                "old": fixture["wcc"].get(name, {}).get("title"),
                "new": simulator["wcc"]["title_probability"][name],
            }
            for name in simulator["wcc"]["title_probability"]
        },
    }


def _validate(
    root: Path,
    race: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    cutoff: datetime,
    model: ContractModel,
    pre_model: ContractModel,
    events: list[EventSimulation],
    model_sha: str,
    run_id: str,
    artifacts: list[Path],
) -> dict[str, bool]:
    """Fail closed on coherence, leakage, provenance and labelling."""
    _check_race(race)
    for row in rows:
        require_known_by(row["feature_timestamp"], cutoff)
        if any(name.startswith("label_") for name in row):
            raise ValueError("live rows contain outcome labels")
    for owner in (model, pre_model):
        for row in owner.training:
            require_known_by(row["label_available_at"], cutoff)
            if row["label_dnf"] is not None and row["label_dnf_available_at"] > cutoff:
                raise ValueError("DNF training label published after the cutoff")
    if any(
        event.source_hash != model_sha or not event.model_id.startswith(run_id) for event in events
    ):
        raise ValueError("championship uses race samples outside this development run")
    for path in artifacts:
        if path.suffix == ".parquet":
            labels = set(pq.read_table(path)["validation_status"].to_pylist())
        else:
            labels = {json.loads(path.read_text(encoding="utf-8"))["validation_status"]}
        if labels != {STATUS}:
            raise ValueError(f"{path.name} is not labelled development_only")
    return {
        "race_probabilities_coherent": True,
        "snapshot_has_no_post_cutoff_data": True,
        "training_labels_known_at_cutoff": True,
        "training_uses_immutable_dataset_manifests": True,
        "championship_uses_only_development_race_samples": True,
        "outputs_labelled_development_only": True,
    }
