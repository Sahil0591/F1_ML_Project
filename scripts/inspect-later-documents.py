"""Retain and summarize FIA documents published after a race final classification."""

import argparse
import json
from pathlib import Path

import httpx

from f1_ml_predictor.trust.historical import _retain_response, inspect_pdf
from f1_ml_predictor.trust.winter import _version_key, registry_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int)
    parser.add_argument("rounds", type=int, nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalogs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (root / "data/benchmarks/historical_audit").glob("candidates-*.json")
    ]
    matches = [
        catalog
        for catalog in catalogs
        if len(catalog["candidates"]) == 107
        and all(row["status"] == "audit_required" for row in catalog["candidates"])
    ]
    if len(matches) != 1:
        raise ValueError("one exhaustive 107-candidate discovery catalog is required")
    discovery = matches[0]
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        for round_number in args.rounds:
            candidate = next(
                row
                for row in discovery["candidates"]
                if row["season"] == args.season and row["round"] == round_number
            )
            registry = _retain_response(root, client.get(candidate["index_url"]))
            rows = registry_rows((root / registry["path"]).read_text(encoding="utf-8"))
            finals = [
                row
                for row in rows
                if "final race classification" in row["title"].lower()
                and row.get("url")
                and not row["recalled"]
            ]
            final = max(finals, key=_version_key)
            print(f"{args.season} round {round_number}: final {final['publication_cet']}")
            for row in sorted(rows, key=_version_key):
                if (
                    not row.get("url")
                    or row["recalled"]
                    or _version_key(row) <= _version_key(final)
                    or row["title"].lower() == "championship points"
                ):
                    continue
                artifact = _retain_response(root, client.get(row["url"]))
                inspected = inspect_pdf(root / artifact["path"])
                normalized = " ".join(inspected["text"].split())
                print(
                    json.dumps(
                        {
                            "document_id": row["document_id"] or inspected["document_id"],
                            "title": row["title"],
                            "publication_cet": row["publication_cet"],
                            "url": row["url"],
                            "sha256": artifact["sha256"],
                            "path": artifact["path"],
                            "first_text": normalized[:220],
                            "last_text": normalized[-260:],
                        },
                        ensure_ascii=True,
                    )
                )


if __name__ == "__main__":
    main()
