"""Create cumulative diagnostic feature masks over one frozen Gold cohort."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data/benchmarks/gold_core_rolling_v1"
    / "4c0cae1bc87191ef5e3b78eed2b6228396e4efec7ae8e7028651712e53de5412"
)
OUTPUT = ROOT / "data/benchmarks/gold_diagnostic_ablations_v1"
QUALIFYING = {
    "qualifying_position",
    "qualifying_last_session_seconds",
    "grid_position",
}
RECENT_FORM = {
    "recent_finish_mean",
    "recent_dnf_rate",
    *(f"recent_finish_mean_{window}" for window in (3, 5, 10)),
    *(f"recent_dnf_rate_{window}" for window in (3, 5, 10)),
    *(f"history_count_{window}" for window in (3, 5, 10)),
}
CONSTRUCTOR_FORM = {"constructor_recent_classification_mean", "recent_pit_stop_seconds"}
TEAMMATE = {"teammate_qualifying_position_delta"}
CHAMPIONSHIP = {
    "driver_championship_points",
    "constructor_championship_points",
    "circuit_length_km",
    "is_street_circuit",
}
PRACTICE = {
    "practice_observed_best_lap_seconds",
    "practice_summary_mean_lap_seconds",
    "practice_summary_mean_tyre_age",
    "practice_observed_compound_count",
}
WEATHER = {
    "forecast_temperature_2m",
    "forecast_precipitation_probability",
    "forecast_wind_speed_10m",
}
STAGES = (
    ("qualifying_only", QUALIFYING),
    ("qualifying_plus_recent_form", RECENT_FORM),
    ("plus_constructor_form", CONSTRUCTOR_FORM),
    ("plus_teammate_features", TEAMMATE),
    ("plus_championship_context", CHAMPIONSHIP),
    ("plus_practice_where_available", PRACTICE),
    ("plus_weather_where_available", WEATHER),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    benchmark = args.source
    manifest = json.loads((benchmark / "manifest.json").read_text())
    source = benchmark / "gold.parquet"
    if digest(source) != manifest["datasets"]["Gold"]["sha256"]:
        raise ValueError("source Gold benchmark changed")
    table = pq.read_table(source)
    active = set()
    for name, group in STAGES:
        active.update(group)
        destination = OUTPUT / digest(source) / name
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        for source_row in table.to_pylist():
            row = source_row.copy()
            for feature in manifest["feature_columns"]:
                base = feature.removesuffix("_missing")
                if base not in active:
                    row[feature] = True if feature.endswith("_missing") else None
            rows.append(row)
        target = destination / "gold.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target, compression="zstd")
        datasets = dict(manifest["datasets"])
        datasets["Gold"] = {**datasets["Gold"], "sha256": digest(target)}
        for tier in ("Silver", "Development"):
            secondary = benchmark / str(datasets[tier]["path"])
            copy = destination / secondary.name
            shutil.copyfile(secondary, copy)
            if digest(copy) != datasets[tier]["sha256"]:
                raise ValueError("secondary benchmark hash mismatch")
        shutil.copyfile(benchmark / "coverage.json", destination / "coverage.json")
        if digest(destination / "coverage.json") != manifest["coverage_sha256"]:
            raise ValueError("coverage hash mismatch")
        ablated = {
            **manifest,
            "datasets": datasets,
            "diagnostic_ablation": name,
            "diagnostic_source_manifest_sha256": digest(benchmark / "manifest.json"),
            "active_feature_groups": [
                stage for stage, _ in STAGES[: STAGES.index((name, group)) + 1]
            ],
            "selection_eligible": False,
        }
        (destination / "manifest.json").write_text(
            json.dumps(ablated, sort_keys=True), encoding="utf-8"
        )
        print(destination.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
