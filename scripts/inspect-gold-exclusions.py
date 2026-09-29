"""Inspect retained FIA registry and PDF evidence for excluded Gold candidates."""

import json
from pathlib import Path

import httpx

from f1_ml_predictor.trust.historical import _retain_response, inspect_pdf
from f1_ml_predictor.trust.winter import _publication, registry_rows

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/HISTORICAL_EXPANSION_CANDIDATES.json"


def main() -> None:
    candidates = json.loads(CATALOG.read_text(encoding="utf-8"))["candidates"]
    for round_number in (9, 11, 15):
        item = next(
            row for row in candidates if row["season"] == 2025 and row["round"] == round_number
        )
        artifact = item["research_registry_artifact"]
        rows = registry_rows((ROOT / artifact["path"]).read_text(encoding="utf-8"))
        final = next(row for row in rows if row["url"] == item["final_race_label"]["url"])
        print(
            f"round {round_number}: final {final['document_id']} {_publication(final).isoformat()}"
        )
        for row in rows:
            if _publication(row) >= _publication(final) and row != final:
                print(row["document_id"], row["title"], row["url"], row["recalled"])
        if round_number == 11:
            print("qualifying URL:", item["qualifying_url"])
            pdfs = list((ROOT / "data/raw/fia_audit/requests").glob("*.json"))
            for metadata_path in pdfs:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata["url"] in {item["qualifying_url"], final["url"]}:
                    extracted = inspect_pdf(ROOT / metadata["path"])
                    print(metadata["url"], "text:", extracted["text"][:3500])
        if round_number == 15:
            decision = next(row for row in rows if row["title"].startswith("Decision - Williams"))
            with httpx.Client(timeout=30, follow_redirects=False) as client:
                artifact = _retain_response(ROOT, client.get(decision["url"]))
            print("decision artifact:", artifact)
            print("right of review decision:", inspect_pdf(ROOT / artifact["path"])["text"][-5500:])


if __name__ == "__main__":
    main()
