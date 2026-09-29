"""Inspect registered FIA final PDFs for explicit retirement-status evidence."""

import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file
from f1_ml_predictor.trust.historical import inspect_pdf

ROOT = Path(__file__).resolve().parents[1]
STATUS = re.compile(r"\b(?:RETIRED|FINISHED|DNF|DNS|DSQ|NOT CLASSIFIED)\b", re.I)


def main() -> None:
    registry = json.loads((ROOT / "data/benchmarks/gold_core_registry.json").read_text())
    categories: Counter[str] = Counter()
    reports = []
    for race in registry["races"]:
        outcome = race["outcomes"]
        table = pq.read_table(_safe_file(ROOT, outcome["path"], outcome["sha256"]))
        rows = table.to_pylist()
        categories.update(row["dnf_category"] for row in rows)
        target_dir = (
            ROOT / "data/features/historical_evidence" / race["features"]["sha256"] / "targets"
        )
        target_path = target_dir / f"{outcome['sha256']}.json"
        target = json.loads(target_path.read_text())
        binding = target["document_bindings"][0]
        pdf = _safe_file(ROOT, binding["path"], binding["sha256"])
        text = inspect_pdf(pdf)["text"]
        reports.append(
            {
                "event_id": race["event_id"],
                "driver_race_observations": len(rows),
                "audited_dnf_labels": sum(row["dnf"] is not None for row in rows),
                "pdf_status_tokens": sorted({match[0].upper() for match in STATUS.finditer(text)}),
                "document_sha256": binding["sha256"],
            }
        )
    print(json.dumps({"categories": categories, "races": reports}, indent=2))


if __name__ == "__main__":
    main()
