"""Exact-byte, visually audited transcription of damaged qualifying rows."""

import re
from typing import Any

import pyarrow as pa

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.fia_tables import QUALIFYING_SCHEMA, parse_qualifying_text


def reviewed_qualifying(
    text: str,
    event: EventId,
    drivers: dict[str, str],
    constructors: dict[str, str],
    review: dict[str, Any],
    document_sha256: str,
) -> pa.Table:
    """Use declared source transcriptions only for the exact reviewed PDF.

    A car number checks the source row; it never resolves driver identity.
    Q1-only damaged rows are supported. Stage membership must have been checked
    visually, and every transcribed time must still occur literally in the row.
    All other rows use the ordinary parser. No fuzzy names or inferred times.
    """
    if (
        review.get("document_sha256") != document_sha256
        or review.get("event_id") != event.partition()
        or review.get("visually_audited") is not True
        or not review.get("audit_reference")
        or not review.get("page_number")
    ):
        raise ValueError("transcription needs an exact PDF hash and completed visual audit")
    lines = text.splitlines()
    transcribed = []
    for item in review["rows"]:
        matching = [index for index, line in enumerate(lines) if item["source_fragment"] in line]
        if len(matching) != 1:
            raise ValueError("transcription source fragment is missing or ambiguous")
        index = matching[0]
        original = lines[index]
        prefix = re.match(r"^[\s\x03]*(\d+)[\s\x03]+(\d+)[\s\x03]+", original)
        times = re.findall(r"(?<![\d:])\d{1,2}:[0-5]\d\.\d{3}(?![\d:])", original)
        if (
            prefix is None
            or int(prefix[1]) != item["position"]
            or int(prefix[2]) != item["car_number"]
            or times != [item["q1"]]
            or item.get("q2") is not None
            or item.get("q3") is not None
        ):
            raise ValueError("transcription contradicts the retained row or supported Q1 stage")
        minutes, seconds = item["q1"].split(":")
        transcribed.append(
            {
                "event_id": event.partition(),
                "driver_id": drivers[item["driver_display"]],
                "constructor_id": constructors[item["constructor_display"]],
                "position": item["position"],
                "q1_seconds": int(minutes) * 60 + float(seconds),
                "q2_seconds": None,
                "q3_seconds": None,
            }
        )
        lines[index] = ""
    parsed = parse_qualifying_text("\n".join(lines), event, drivers, constructors).to_pylist()
    rows = [*parsed, *transcribed]
    if len({row["driver_id"] for row in rows}) != len(rows):
        raise ValueError("transcription duplicates a parsed canonical driver")
    positions = [row["position"] for row in rows if row["position"] is not None]
    if len(set(positions)) != len(positions):
        raise ValueError("transcription duplicates a parsed classification position")
    return pa.Table.from_pylist(
        sorted(rows, key=lambda row: row["driver_id"]), schema=QUALIFYING_SCHEMA
    )
