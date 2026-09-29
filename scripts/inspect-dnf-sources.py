"""Summarize retained binary DNF cross-checks without assigning labels."""

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT
    / "data/raw/dnf_audit_v1"
    / "capture-4a809a21c40d5f65c86d1bd6bc420471119a33578228fa597379c0725e604a89.json"
)


def payload(record: dict) -> dict | list:
    return json.loads(_safe_file(ROOT, record["path"], record["sha256"]).read_bytes())


def main() -> None:
    registry = json.loads((ROOT / "data/benchmarks/gold_core_registry.json").read_text())
    races = {row["event_id"]: row for row in registry["races"]}
    capture = json.loads(CAPTURE.read_text())
    counts: Counter[str] = Counter()
    examples: dict[str, list] = {}
    for event in capture["races"]:
        if event["status"] != "captured":
            counts["capture_failed"] += 1
            continue
        source = payload(event["jolpica"])["MRData"]["RaceTable"]["Races"][0]["Results"]
        numbers = {int(row["Driver"]["permanentNumber"]): row for row in source}
        openf1 = payload(event["openf1_session_result"])
        final = pq.read_table(
            _safe_file(
                ROOT,
                races[event["event_id"]]["outcomes"]["path"],
                races[event["event_id"]]["outcomes"]["sha256"],
            )
        ).to_pylist()
        identities = {row["driver_id"]: row for row in final}
        seen = set()
        for row in openf1:
            number = row["driver_number"]
            corresponding = numbers.get(number)
            if corresponding is None:
                counts["missing_jolpica_number"] += 1
                continue
            driver = corresponding["Driver"]["driverId"]
            seen.add(driver)
            fia = identities.get(driver)
            if fia is None:
                counts["missing_fia_driver"] += 1
                continue
            key = (
                f"openf1={row.get('dnf')},{row.get('dns')},{row.get('dsq')} "
                f"jolpica={corresponding['status']} "
                f"fia={fia['raw_status']}:{fia['dnf_category']}"
            )
            counts[key] += 1
            examples.setdefault(key, []).append((event["event_id"], driver))
        counts["missing_openf1_drivers"] += len(set(identities) - seen)
    for key, count in counts.most_common():
        print(count, key, examples.get(key, [])[:2])


if __name__ == "__main__":
    main()
