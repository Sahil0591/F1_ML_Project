"""Export verified development prediction runs as display-ready JSON for the web app.

The exporter only reads immutable run directories under
``data/predictions/development/next_race``. It checks every artifact hash recorded in
the run manifest, validates the probability contract, and writes small typed JSON
documents. No prediction is recomputed: race and championship values are copied
from the artifacts unchanged. The few derived fields (championship top 3 from the
final position distribution, the most likely final position, the projected order
by mean final points, and comparisons with actual results) are documented in
``docs/WEB_FRONTEND.md``.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.ingestion.cache import RawCache
from f1_ml_predictor.prediction.live_features import load_schedule
from f1_ml_predictor.prediction.protocol import CUTOFFS
from f1_ml_predictor.prediction.season import published_standings
from f1_ml_predictor.prediction.sprint import SPRINT_CONTRACT
from f1_ml_predictor.scoring.ledger import load_scoring_ledger

SCHEMA_VERSION = 1
SUPPORTED_METHODOLOGY = "cutoff-specific-v3"
RUNS = Path("data/predictions/development/next_race")
DEFAULT_OUTPUT = Path("web/public/data")
_TOLERANCE = 1e-6
# Weekend order: the sprint snapshot follows practice and precedes main qualifying.
EXPORT_CUTOFFS = (*CUTOFFS[:2], SPRINT_CONTRACT, *CUTOFFS[2:])


def session_of(cutoff: str) -> str:
    return "sprint" if cutoff == SPRINT_CONTRACT else "race"


class ExportError(ValueError):
    """A run artifact failed verification and cannot be published."""


@dataclass(frozen=True)
class Run:
    directory: Path
    season: int
    round: int
    cutoff: str
    manifest: dict[str, Any]
    race: dict[str, Any]
    championship: dict[str, Any]
    predictions: list[dict[str, Any]]

    @property
    def run_id(self) -> str:
        return str(self.manifest["model_run_id"])

    @property
    def timestamp(self) -> str:
        return str(self.manifest["prediction_timestamp_utc"])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"{path.name} is not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def _probability(value: Any, name: str) -> float:
    _require(
        isinstance(value, int | float) and not isinstance(value, bool), f"{name} is not a number"
    )
    _require(math.isfinite(value) and 0.0 <= value <= 1.0, f"{name} is outside [0, 1]")
    return float(value)


def _partition(directory: Path, runs_root: Path) -> tuple[int, int, str]:
    parts = directory.relative_to(runs_root).parts
    _require(len(parts) == 4, f"unexpected run layout {directory}")
    season, round_, cutoff, _ = parts
    _require(season.startswith("season=") and round_.startswith("round="), "bad partition")
    return int(season[7:]), int(round_[6:]), cutoff


def load_run(directory: Path, runs_root: Path) -> Run:
    """Read one run, verify manifest hashes and validate the race and title contracts."""
    season, round_, cutoff = _partition(directory, runs_root)
    manifest = _read_json(directory / "manifest.json")
    _require(
        "methodology" in manifest,
        "legacy c52b674 artifact without a methodology field; only "
        f"{SUPPORTED_METHODOLOGY} runs are exported",
    )
    _require(
        manifest.get("methodology") == SUPPORTED_METHODOLOGY,
        f"methodology {manifest.get('methodology')!r} is not {SUPPORTED_METHODOLOGY}",
    )
    _require(manifest.get("validation_status") == "development_only", "not development_only")
    _require(manifest.get("validated_forecast") is False, "validated_forecast must be false")
    _require(manifest.get("cutoff_kind") == cutoff, "cutoff differs from the run directory")
    _require(cutoff in EXPORT_CUTOFFS, f"unknown cutoff {cutoff}")
    _require(
        manifest.get("session", "race") == session_of(cutoff),
        "manifest session differs from its cutoff",
    )
    artifacts = manifest["artifacts"]
    for name, path in (
        ("predictions", directory / "predictions.parquet"),
        ("race_distribution", directory / "race_distribution.json"),
        ("championship", directory / "championship.json"),
    ):
        _require(path.exists(), f"missing {path.name}")
        _require(file_sha256(path) == artifacts[name]["sha256"], f"{path.name} hash mismatch")
    race = _read_json(directory / "race_distribution.json")
    championship = _read_json(directory / "championship.json")
    _require(
        championship.get("race_distribution_sha256") == artifacts["race_distribution"]["sha256"],
        "championship is not bound to this race distribution",
    )
    for document in (race, championship):
        _require(document.get("model_run_id") == manifest["model_run_id"], "run ID mismatch")
        _require(document.get("validation_status") == "development_only", "not development_only")
    predictions = pq.read_table(directory / "predictions.parquet").to_pylist()
    run = Run(directory, season, round_, cutoff, manifest, race, championship, predictions)
    _validate_race(run)
    _validate_championship(championship)
    return run


def _validate_race(run: Run) -> None:
    drivers = run.race["drivers"]
    count = len(drivers)
    _require(count >= 2, "race needs at least two drivers")
    ids = [item["driver_id"] for item in drivers]
    _require(len(set(ids)) == count, "duplicate driver IDs")
    for item in drivers:
        name = item["driver_id"]
        for key in ("winner_probability", "podium_probability", "dnf_model_probability"):
            _probability(item[key], f"{name} {key}")
        finish = [_probability(value, f"{name} finish") for value in item["finish_distribution"]]
        _require(len(finish) == count, f"{name} finish distribution length")
        _require(abs(sum(finish) - 1.0) < _TOLERANCE, f"{name} finish distribution sum")
        _require(abs(finish[0] - item["winner_probability"]) < _TOLERANCE, f"{name} win")
        _require(abs(sum(finish[:3]) - item["podium_probability"]) < _TOLERANCE, f"{name} podium")
        _require(1.0 <= item["expected_position"] <= count, f"{name} expected position")
        low, high = item["position_interval_80"]
        _require(1 <= low <= high <= count, f"{name} 80% interval")
        if "clean_expected_position" in item:
            _require(1.0 <= item["clean_expected_position"] <= count, f"{name} clean expected")
    positions = [item.get("predicted_position") for item in drivers]
    if any(value is not None for value in positions):
        _require(
            sorted(value for value in positions if value is not None) == list(range(1, count + 1)),
            "predicted_position must be a permutation of 1..n",
        )
    by_driver = {row["driver_id"]: row for row in run.predictions}
    _require(set(by_driver) == set(ids), "predictions.parquet drivers differ")
    for item in drivers:
        row = by_driver[item["driver_id"]]
        for key in (
            "constructor_id",
            "winner_probability",
            "podium_probability",
            "dnf_model_probability",
            "expected_position",
            "most_likely_position",
            "predicted_position",
            "clean_expected_position",
            "clean_most_likely_position",
        ):
            _require(row.get(key) == item.get(key), f"{item['driver_id']} {key} differs in parquet")


def _validate_championship(championship: dict[str, Any]) -> None:
    simulator = championship["simulator"]
    points = championship["starting_points"]
    for key, start in (("wdc", points["driver_points"]), ("wcc", points["constructor_points"])):
        result = simulator[key]
        names = set(result["title_probability"])
        _require(bool(names), f"{key} has no entries")
        _require(set(result["mean_final_points"]) == names, f"{key} entries differ")
        _require(set(result["final_position_probability"]) == names, f"{key} entries differ")
        _probability(result["unresolved_tie_probability"], f"{key} unresolved tie")
        for name in names:
            _probability(result["title_probability"][name], f"{key} {name} title")
            places = result["final_position_probability"][name]
            _require(len(places) == len(names), f"{key} {name} position distribution length")
            for value in places:
                _probability(value, f"{key} {name} final position")
            _require(abs(sum(places) - 1.0) < _TOLERANCE, f"{key} {name} positions sum")
            _require(
                result["mean_final_points"][name] >= start.get(name, 0.0) - _TOLERANCE,
                f"{key} {name} loses points",
            )


def discover_runs(root: Path) -> tuple[list[Run], list[dict[str, str]]]:
    """Load every supported run; unsupported runs are reported, never silently mixed in."""
    runs_root = root / RUNS
    runs: list[Run] = []
    excluded: list[dict[str, str]] = []
    for manifest in sorted(runs_root.glob("season=*/round=*/*/*/manifest.json")):
        directory = manifest.parent
        relative = directory.relative_to(root).as_posix()
        try:
            runs.append(load_run(directory, runs_root))
        except (ExportError, KeyError, TypeError) as exc:
            excluded.append({"path": relative, "reason": str(exc)})
    return runs, excluded


class Names:
    """Display names from the retained Jolpica result and qualifying caches."""

    def __init__(self, root: Path) -> None:
        self.cache = RawCache(root / "data/raw/jolpica")
        self.drivers: dict[str, dict[str, str]] = {}
        self.constructors: dict[str, str] = {}
        self.seasons: set[int] = set()

    def load(self, season: int, rounds: Iterable[int]) -> None:
        if season in self.seasons:
            return
        self.seasons.add(season)
        for number in rounds:
            for collection in ("results", "qualifying"):
                try:
                    cached = self.cache.load(f"season={season}/round={number:02d}/{collection}")
                except ValueError:
                    cached = None
                if cached is None:
                    continue
                for race in cached.items:
                    for row in race.get("Results", []) + race.get("QualifyingResults", []):
                        driver, team = row.get("Driver", {}), row.get("Constructor", {})
                        if driver.get("driverId"):
                            self.drivers[driver["driverId"]] = {
                                "name": f"{driver.get('givenName', '')} "
                                f"{driver.get('familyName', '')}".strip(),
                                "code": driver.get("code", ""),
                            }
                        if team.get("constructorId") and team.get("name"):
                            self.constructors[team["constructorId"]] = team["name"]

    @staticmethod
    def _fallback(identifier: str) -> str:
        return identifier.replace("_", " ").title()

    def driver(self, identifier: str) -> str:
        found = self.drivers.get(identifier, {}).get("name")
        return found or self._fallback(identifier)

    def code(self, identifier: str) -> str:
        found = self.drivers.get(identifier, {}).get("code")
        return found or identifier.replace("_", "")[:3].upper()

    def constructor(self, identifier: str) -> str:
        return self.constructors.get(identifier) or self._fallback(identifier)


def _schedule(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Races of the hash-verified schedule observation the run itself used."""
    path = manifest["schedule_source"]["path"]
    load_schedule(root, path)
    record = _read_json(root / path)
    races = record["payload"]["MRData"]["RaceTable"]["Races"]
    return [race for race in races if isinstance(race, dict)]


def _race_meta(schedule: list[dict[str, Any]], round_: int) -> dict[str, Any]:
    for race in schedule:
        if int(race["round"]) == round_:
            location = race.get("Circuit", {}).get("Location", {})
            return {"locality": location.get("locality"), "country": location.get("country")}
    return {"locality": None, "country": None}


def _actual_result(
    root: Path, season: int, round_: int, session: str = "race"
) -> dict[str, Any] | None:
    """Race or sprint classification from the existing Jolpica ingestion, if published."""
    name = "sprint.parquet" if session == "sprint" else "results.parquet"
    path = root / f"data/normalized/season={season}/round={round_:02d}/{name}"
    if not path.exists():
        return None
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        return None
    metadata = table.schema.metadata or {}
    source = metadata.get(b"source_sha256", b"").decode()
    return {
        "source_path": path.relative_to(root).as_posix(),
        "source_sha256": source or None,
        "rows": rows,
    }


def _classified(row: dict[str, Any]) -> bool:
    return str(row.get("position_text") or "").isdigit()


def _comparison(
    drivers: list[dict[str, Any]], actual: dict[str, Any], names: Names
) -> dict[str, Any]:
    predicted = {item["driver_id"]: item for item in drivers}
    classification = []
    for row in sorted(
        actual["rows"], key=lambda item: (item["position"] is None, item["position"])
    ):
        driver = row["driver_id"]
        item = predicted.get(driver)
        classified = _classified(row)
        position = row["position"] if classified else None
        pred = None if item is None else item.get("predicted_position")
        clean = None if item is None else item.get("clean_expected_position")
        classification.append(
            {
                "position": row["position"],
                "position_text": row["position_text"],
                "classified": classified,
                "driver_id": driver,
                "driver_name": names.driver(driver),
                "driver_code": names.code(driver),
                "constructor_id": row["constructor_id"],
                "constructor_name": names.constructor(row["constructor_id"] or ""),
                "grid": row["grid"],
                "laps": row["laps"],
                "points": row["points"],
                "status": row["status"],
                "predicted_position": pred,
                "position_delta": None if position is None or pred is None else pred - position,
                "clean_expected_position": clean,
                "clean_expected_error": None
                if position is None or clean is None
                else position - clean,
                "winner_probability": None if item is None else item["winner_probability"],
                "podium_probability": None if item is None else item["podium_probability"],
                "dnf_model_probability": None if item is None else item["dnf_model_probability"],
            }
        )
    finishers = [row for row in classification if row["classified"]]
    winner = next((row for row in finishers if row["position"] == 1), None)
    podium = [row for row in finishers if row["position"] in (1, 2, 3)]
    retired = [row for row in classification if not row["classified"]]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    predicted_winner = next(
        (item["driver_id"] for item in drivers if item.get("predicted_position") == 1), None
    )
    return {
        "classification": classification,
        "summary": {
            "winner_id": None if winner is None else winner["driver_id"],
            "winner_probability": None if winner is None else winner["winner_probability"],
            "predicted_winner_id": predicted_winner,
            "podium": [
                {"driver_id": row["driver_id"], "podium_probability": row["podium_probability"]}
                for row in podium
            ],
            "retirements": [
                {
                    "driver_id": row["driver_id"],
                    "status": row["status"],
                    "dnf_model_probability": row["dnf_model_probability"],
                }
                for row in retired
            ],
            "predicted_position_mae": mean(
                [
                    abs(row["position_delta"])
                    for row in finishers
                    if row["position_delta"] is not None
                ]
            ),
            "clean_expected_mae": mean(
                [
                    abs(row["clean_expected_error"])
                    for row in finishers
                    if row["clean_expected_error"] is not None
                ]
            ),
            "exact_position_hits": sum(1 for row in finishers if row["position_delta"] == 0),
            "classified_finishers": len(finishers),
        },
    }


def _ranked(values: dict[str, float], title: dict[str, float]) -> list[str]:
    """Projected order: mean final points, then title probability, then identifier."""
    return sorted(values, key=lambda name: (-values[name], -title[name], name))


def _standings(
    result: dict[str, Any],
    start: dict[str, float],
    fixed: dict[str, float] | None,
    label: Callable[[str], str],
    team: dict[str, str] | None,
    names: Names,
) -> list[dict[str, Any]]:
    entries = []
    order = _ranked(result["mean_final_points"], result["title_probability"])
    for projected, name in enumerate(order, 1):
        places = result["final_position_probability"][name]
        constructor = None if team is None else team.get(name)
        entries.append(
            {
                "id": name,
                "name": label(name),
                "code": names.code(name) if team is not None else None,
                "constructor_id": constructor,
                "constructor_name": None if constructor is None else names.constructor(constructor),
                "points_now": start.get(name, 0.0),
                "expected_final_points": result["mean_final_points"][name],
                "title_probability": result["title_probability"][name],
                "tied_for_title_probability": result["tied_for_title_probability"].get(name),
                "top3_probability": sum(places[:3]),
                "most_likely_position": max(range(len(places)), key=lambda i: places[i]) + 1,
                "final_position_probability": places,
                "projected_position": projected,
                "fixed_strength_title_probability": None if fixed is None else fixed.get(name),
            }
        )
    return entries


def _championship(run: Run, names: Names) -> dict[str, Any]:
    payload = run.championship
    simulator = payload["simulator"]
    uncertainty = payload["uncertainty"]
    start = payload["starting_points"]
    fixed = uncertainty.get("sensitivity", {}).get("fixed_strength_no_persistent_uncertainty")
    teams = {item["driver_id"]: item["constructor_id"] for item in run.race["drivers"]}
    sensitivity = {
        key.removeprefix("single_candidate_"): value
        for key, value in uncertainty.get("sensitivity", {}).items()
        if key.startswith("single_candidate_")
    }
    return {
        "season": simulator["season"],
        "status": simulator["status"],
        "simulations": simulator["simulations"],
        "worlds": uncertainty["worlds"],
        "orders_per_world": uncertainty["orders_per_world"],
        "monte_carlo_standard_error_max": uncertainty["monte_carlo_standard_error_max"],
        "standings_available_at": start["available_at"],
        "standings_notes": payload["standings_source"]["notes"],
        "remaining_sessions": [
            {
                "event_id": item["event_id"],
                "race_name": item["race_name"],
                "session": item["session"],
                "scheduled_at": item["scheduled_at"],
            }
            for item in payload["sessions"]
        ],
        "assumptions": payload["assumptions"],
        "limitations": simulator["limitations"],
        "candidate_sensitivity": sensitivity,
        "wdc": {
            "unresolved_tie_probability": simulator["wdc"]["unresolved_tie_probability"],
            "entries": _standings(
                simulator["wdc"],
                start["driver_points"],
                None if fixed is None else fixed["wdc"],
                names.driver,
                teams,
                names,
            ),
        },
        "wcc": {
            "unresolved_tie_probability": simulator["wcc"]["unresolved_tie_probability"],
            "entries": _standings(
                simulator["wcc"],
                start["constructor_points"],
                None if fixed is None else fixed["wcc"],
                names.constructor,
                None,
                names,
            ),
        },
    }


def _historical(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    path = root / manifest["evaluation"]["live_masked_evaluation"]
    if not path.exists():
        return None
    evaluation = _read_json(path)
    primary = manifest["evaluation"]["primary"]

    def metrics(values: dict[str, Any]) -> dict[str, Any]:
        return {
            "winner_log_loss": values["winner"]["log_loss"],
            "winner_brier": values["winner"]["brier_score"],
            "top1_accuracy": values["winner"]["top_1_accuracy"],
            "podium_brier": values["podium"]["brier_score"],
            "finish_mae": values["finishing_position"]["mean_absolute_error"],
        }

    return {
        "outer_races": evaluation["outer_races"],
        "primary": metrics(evaluation["models"][primary]["metrics"]),
        "baselines": {
            name: metrics(values) for name, values in sorted(evaluation["baselines"].items())
        },
        "formal_gate": {
            task: value["status"]
            for task, value in evaluation["models"][primary]["formal_gate"].items()
        },
    }


def _driver_rows(run: Run, names: Names) -> list[dict[str, Any]]:
    rows = []
    for item in run.race["drivers"]:
        rows.append(
            {
                "driver_id": item["driver_id"],
                "driver_name": names.driver(item["driver_id"]),
                "driver_code": names.code(item["driver_id"]),
                "constructor_id": item["constructor_id"],
                "constructor_name": names.constructor(item["constructor_id"]),
                "predicted_position": item.get("predicted_position"),
                "winner_probability": item["winner_probability"],
                "podium_probability": item["podium_probability"],
                "dnf_probability": item["dnf_probability"],
                "dnf_model_probability": item["dnf_model_probability"],
                "expected_position": item["expected_position"],
                "most_likely_position": item["most_likely_position"],
                "clean_expected_position": item.get("clean_expected_position"),
                "clean_most_likely_position": item.get("clean_most_likely_position"),
                "position_interval_80": list(item["position_interval_80"]),
                "winner_draws": item["winner_draws"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["predicted_position"] is None,
            row["predicted_position"] or 0,
            row["expected_position"],
        ),
    )


def snapshot(root: Path, run: Run, names: Names, schedule: list[dict[str, Any]]) -> dict[str, Any]:
    manifest, race = run.manifest, run.race
    event = manifest["event"]
    drivers = _driver_rows(run, names)
    ood = manifest["ood"]
    dnf_values = {round(item["dnf_model_probability"], 6) for item in drivers}
    session = session_of(run.cutoff)
    actual = _actual_result(root, run.season, run.round, session)
    dataset = manifest["dataset"]
    session_start = event.get("sprint_start") if session == "sprint" else event["race_start"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "race_snapshot",
        "identity": {
            "season": run.season,
            "round": run.round,
            "cutoff": run.cutoff,
            "session": session,
            "run_id": run.run_id,
            "methodology": manifest["methodology"],
            "validation_status": manifest["validation_status"],
            "development_only": manifest["validation_status"] == "development_only",
            "validated_forecast": manifest["validated_forecast"],
            "prediction_timestamp": manifest["prediction_timestamp_utc"],
            "created_at": manifest["created_at"],
            # A replay at an earlier cutoff is honest only if the page says when it was made.
            "generated_after_race_start": datetime.fromisoformat(manifest["created_at"])
            > datetime.fromisoformat(session_start or event["race_start"]),
            "git_commit": manifest.get("git_commit"),
            "source_directory": run.directory.relative_to(root).as_posix(),
        },
        "warning": manifest["warning"],
        "event": {
            "event_id": event["event_id"],
            "race_name": event["race_name"],
            "circuit_id": event["circuit_id"],
            "circuit_name": event["circuit_name"],
            **_race_meta(schedule, run.round),
            "race_start": event["race_start"],
            "first_practice": event["first_practice"],
            "qualifying_start": event["qualifying_start"],
            "sprint_weekend": event["sprint_weekend"],
            "sprint_qualifying_start": event.get("sprint_qualifying_start"),
            "sprint_start": event.get("sprint_start"),
        },
        "status": {
            "ood_status": ood["status"],
            "ood_reasons": ood["reasons"],
            "unseen_circuit": ood["circuit"]["unseen_in_training"],
            "training_circuits": ood["circuit"]["training_circuits"],
            "development_quality": manifest["development_quality"],
            "checks_passed": all(manifest["validation_checks"].values()),
            "notes": manifest["notes"],
        },
        "race": {
            "draws": race["draws"],
            "seed": race["seed"],
            "monte_carlo_resolution": race["monte_carlo_resolution"],
            "predicted_order_available": all(
                row["predicted_position"] is not None for row in drivers
            ),
            "dnf_field_wide": len(dnf_values) == 1,
            "drivers": drivers,
        },
        "championship": _championship(run, names),
        "model": {
            "primary_model": race["primary_model"],
            "primary_members": race["primary_members"],
            "experimental_model": manifest["evaluation"]["experimental_model"],
            "calibration": race["calibration"],
            "dnf_model": manifest["evaluation"]["dnf"]["model"],
            "dnf_candidates": manifest["evaluation"]["dnf"]["candidates"],
            "dnf_known_labels": dataset["dnf_known_labels"],
            "feature_contract": manifest["feature_contract"],
            "protocol_version": manifest["protocol"]["version"],
            "protocol_sha256_short": manifest["protocol"]["sha256"][:12],
            "dataset_version": dataset["gold"],
            "dataset_hash_short": dataset["gold_manifest_sha256"][:12],
            "dnf_dataset_version": dataset["dnf"],
            "masked_for_training": manifest["evaluation"]["masked_for_training"],
            "devices": manifest["execution"]["devices"],
            "sharpness": manifest["sharpness"]["live"],
            "historical": _historical(root, manifest),
        },
        "actual_result": None
        if actual is None
        else {
            "source_path": actual["source_path"],
            "source_sha256": actual["source_sha256"],
            **_comparison(drivers, actual, names),
        },
    }


def _final_standings(
    root: Path, season: int, rounds: int, last_race: datetime | None, now: datetime
) -> dict[str, Any] | None:
    """Audited final points from the scoring ledger, once every round has published points."""
    if last_race is None or last_race >= now:
        return None
    try:
        ledger = load_scoring_ledger(
            root / "data/audit/scoring_rules.json", root / "data/audit/event_points_evidence.json"
        )
        standings, notes, _ = published_standings(ledger, EventId(season, rounds + 1), now, {})
    except (OSError, ValueError, KeyError):
        return None

    def ordered(points: dict[str, float]) -> list[dict[str, Any]]:
        names = sorted(points, key=lambda name: (-points[name], name))
        return [
            {
                "id": name,
                "position": index,
                "points": points[name],
                "tied_on_points": sum(1 for other in points.values() if other == points[name]) > 1,
            }
            for index, name in enumerate(names, 1)
        ]

    return {
        "source": "audited scoring ledger",
        "scoring_ledger_sha256": standings.source_hash,
        "notes": notes,
        "drivers": ordered(dict(standings.driver_points)),
        "constructors": ordered(dict(standings.constructor_points)),
    }


def _season_comparison(
    final: dict[str, Any], latest: dict[str, Any], names: Names
) -> dict[str, Any]:
    out: dict[str, list[dict[str, Any]]] = {}
    for key, table, label in (
        ("drivers", "wdc", names.driver),
        ("constructors", "wcc", names.constructor),
    ):
        projected = {item["id"]: item for item in latest["championship"][table]["entries"]}
        rows = []
        for item in final[key]:
            forecast = projected.get(item["id"])
            rows.append(
                {
                    "id": item["id"],
                    "name": label(item["id"]),
                    "final_position": item["position"],
                    "actual_points": item["points"],
                    "projected_position": None
                    if forecast is None
                    else forecast["projected_position"],
                    "projected_final_points": None
                    if forecast is None
                    else forecast["expected_final_points"],
                    "title_probability": None
                    if forecast is None
                    else forecast["title_probability"],
                    "points_error": None
                    if forecast is None
                    else item["points"] - forecast["expected_final_points"],
                    "position_error": None
                    if forecast is None
                    else forecast["projected_position"] - item["position"],
                }
            )
        out[key] = rows
    return out


def _path(season: int, round_: int, cutoff: str) -> str:
    return f"{season}/round-{round_:02d}/{cutoff}.json"


def build_export(root: Path, *, now: datetime | None = None) -> dict[str, dict[str, Any]]:
    """Return every output document keyed by its path relative to the output directory."""
    root = root.resolve()
    clock = now or datetime.now(UTC)
    runs, excluded = discover_runs(root)
    latest_runs: dict[tuple[int, int, str], Run] = {}
    superseded: dict[tuple[int, int, str], list[str]] = {}
    for run in sorted(runs, key=lambda item: (item.manifest["created_at"], item.run_id)):
        key = (run.season, run.round, run.cutoff)
        if key in latest_runs:
            superseded.setdefault(key, []).append(latest_runs[key].run_id)
        latest_runs[key] = run
    names = Names(root)
    outputs: dict[str, dict[str, Any]] = {}
    snapshots: dict[tuple[int, int, str], dict[str, Any]] = {}
    schedules: dict[int, list[dict[str, Any]]] = {}
    for key, run in sorted(latest_runs.items()):
        schedule = _schedule(root, run.manifest)
        schedules[run.season] = schedule
        names.load(run.season, sorted({int(race["round"]) for race in schedule}))
        document = snapshot(root, run, names, schedule)
        snapshots[key] = document
        outputs[_path(*key)] = document
    seasons = []
    for season in sorted({key[0] for key in snapshots}):
        schedule = schedules[season]
        keys = sorted(
            (key for key in snapshots if key[0] == season),
            key=lambda key: snapshots[key]["identity"]["prediction_timestamp"],
        )
        races = []
        for round_ in sorted({key[1] for key in keys}):
            available = {key[2]: snapshots[key] for key in keys if key[1] == round_}
            first = next(iter(available.values()))
            ordered = [cutoff for cutoff in EXPORT_CUTOFFS if cutoff in available]
            races.append(
                {
                    "season": season,
                    "round": round_,
                    "event_id": first["event"]["event_id"],
                    "race_name": first["event"]["race_name"],
                    "circuit_name": first["event"]["circuit_name"],
                    "locality": first["event"]["locality"],
                    "country": first["event"]["country"],
                    "race_start": first["event"]["race_start"],
                    "actual_result_available": first["actual_result"] is not None,
                    "latest_cutoff": ordered[-1],
                    "cutoffs": [
                        {
                            "cutoff": cutoff,
                            "session": session_of(cutoff),
                            "available": cutoff in available,
                            "path": _path(season, round_, cutoff) if cutoff in available else None,
                            "run_id": available[cutoff]["identity"]["run_id"]
                            if cutoff in available
                            else None,
                            "prediction_timestamp": available[cutoff]["identity"][
                                "prediction_timestamp"
                            ]
                            if cutoff in available
                            else None,
                            "predicted_order_available": available[cutoff]["race"][
                                "predicted_order_available"
                            ]
                            if cutoff in available
                            else False,
                            "superseded_run_ids": superseded.get((season, round_, cutoff), []),
                        }
                        for cutoff in EXPORT_CUTOFFS
                    ],
                }
            )
        starts = [
            datetime.fromisoformat(f"{race['date']}T{race.get('time', '23:59:59Z')}")
            for race in schedule
            if race.get("date")
        ]
        total_rounds = max(int(race["round"]) for race in schedule)
        latest_key = keys[-1]
        latest = snapshots[latest_key]
        final = _final_standings(root, season, total_rounds, max(starts, default=None), clock)
        if final is not None:
            for group, label in (("drivers", names.driver), ("constructors", names.constructor)):
                for item in final[group]:
                    item["name"] = label(item["id"])
        season_doc = {
            "schema_version": SCHEMA_VERSION,
            "kind": "season",
            "season": season,
            "total_rounds": total_rounds,
            "completed": final is not None,
            "latest": {
                "round": latest_key[1],
                "cutoff": latest_key[2],
                "path": _path(*latest_key),
                "prediction_timestamp": latest["identity"]["prediction_timestamp"],
            },
            "snapshots": [
                {
                    "round": key[1],
                    "cutoff": key[2],
                    "path": _path(*key),
                    "run_id": snapshots[key]["identity"]["run_id"],
                    "prediction_timestamp": snapshots[key]["identity"]["prediction_timestamp"],
                    "wdc_title": {
                        item["id"]: item["title_probability"]
                        for item in snapshots[key]["championship"]["wdc"]["entries"]
                    },
                    "wcc_title": {
                        item["id"]: item["title_probability"]
                        for item in snapshots[key]["championship"]["wcc"]["entries"]
                    },
                }
                for key in keys
            ],
            "final_standings": final,
            "final_comparison": None if final is None else _season_comparison(final, latest, names),
        }
        outputs[f"{season}/season.json"] = season_doc
        seasons.append(
            {
                "season": season,
                "total_rounds": total_rounds,
                "season_path": f"{season}/season.json",
                "final_results_available": final is not None,
                "races": races,
            }
        )
    if not snapshots:
        raise ExportError("no supported prediction runs to export")
    latest_key = max(snapshots, key=lambda key: snapshots[key]["identity"]["prediction_timestamp"])
    outputs["index.json"] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "index",
        "cutoffs": list(EXPORT_CUTOFFS),
        "latest": {
            "season": latest_key[0],
            "round": latest_key[1],
            "cutoff": latest_key[2],
            "path": _path(*latest_key),
        },
        "seasons": seasons,
        "excluded_runs": excluded,
    }
    return outputs


def _owned(directory: Path) -> bool:
    if not directory.exists():
        return True
    if not any(directory.iterdir()):
        return True
    index = directory / "index.json"
    if not index.exists():
        return False
    try:
        return bool(json.loads(index.read_text(encoding="utf-8")).get("kind") == "index")
    except (OSError, json.JSONDecodeError):
        return False


def write_export(
    root: Path, output: Path | None = None, *, now: datetime | None = None
) -> dict[str, Any]:
    """Build in a temporary directory, then replace the previous export in one step."""
    root = root.resolve()
    target = (output or root / DEFAULT_OUTPUT).resolve()
    if not _owned(target):
        raise ExportError(f"{target} has files that were not written by this exporter")
    outputs = build_export(root, now=now)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=target.parent))
    try:
        for relative, document in sorted(outputs.items()):
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(document, sort_keys=True, indent=2, allow_nan=False)
            path.write_text(text + "\n", encoding="utf-8", newline="\n")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    index = outputs["index.json"]
    return {
        "output": target.as_posix(),
        "files": len(outputs),
        "latest": index["latest"],
        "excluded_runs": len(index["excluded_runs"]),
    }
