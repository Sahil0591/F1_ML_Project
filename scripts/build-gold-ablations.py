"""Create diagnostic feature masks over one frozen Gold cohort and fold set."""

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
    / "10568218180a38d886a06a701e35ae1c8260dc71376b4caa5b0ba030a30ff402"
)
OUTPUT = ROOT / "data/benchmarks/gold_diagnostic_ablations_v1"
QUALIFYING = {
    "qualifying_position",
    "qualifying_last_session_seconds",
    "grid_position",
    "teammate_qualifying_position_delta",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    source = SOURCE / "gold.parquet"
    if digest(source) != manifest["datasets"]["Gold"]["sha256"]:
        raise ValueError("source Gold benchmark changed")
    table = pq.read_table(source)
    for name in ("no_rolling", "qualifying_only"):
        destination = OUTPUT / digest(source) / name
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        for source_row in table.to_pylist():
            row = source_row.copy()
            for feature in manifest["feature_columns"]:
                base = feature.removesuffix("_missing")
                remove = (
                    base.startswith(("recent_finish_mean_", "recent_dnf_rate_", "history_count_"))
                    if name == "no_rolling"
                    else base not in QUALIFYING
                )
                if remove:
                    row[feature] = True if feature.endswith("_missing") else None
            rows.append(row)
        target = destination / "gold.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target, compression="zstd")
        datasets = dict(manifest["datasets"])
        datasets["Gold"] = {**datasets["Gold"], "sha256": digest(target)}
        for tier in ("Silver", "Development"):
            secondary = SOURCE / str(datasets[tier]["path"])
            copy = destination / secondary.name
            shutil.copyfile(secondary, copy)
            if digest(copy) != datasets[tier]["sha256"]:
                raise ValueError("secondary benchmark hash mismatch")
        shutil.copyfile(SOURCE / "coverage.json", destination / "coverage.json")
        if digest(destination / "coverage.json") != manifest["coverage_sha256"]:
            raise ValueError("coverage hash mismatch")
        ablated = {
            **manifest,
            "datasets": datasets,
            "diagnostic_ablation": name,
            "diagnostic_source_manifest_sha256": digest(SOURCE / "manifest.json"),
            "selection_eligible": False,
        }
        (destination / "manifest.json").write_text(
            json.dumps(ablated, sort_keys=True), encoding="utf-8"
        )
        print(destination.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
