from copy import deepcopy

import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.transcription import reviewed_qualifying

EVENT = EventId(2025, 1)
DRIVERS = {"Known DRIVER": "known", "Reviewed DRIVER": "reviewed"}
CONSTRUCTORS = {"Known Team": "known_team"}
TEXT = (
    "NO DRIVER Q1 Q2 Q3\n1 1 Known DRIVER Known Team 1:15.000\n"
    "16\x03 12\x03 damaged name Known Team 1:16.525 9 16:17:16"
)
REVIEW = {
    "document_sha256": "exact_pdf",
    "event_id": EVENT.partition(),
    "visually_audited": True,
    "audit_reference": "fixture visual transcription",
    "page_number": 2,
    "rows": [
        {
            "source_fragment": "damaged name",
            "position": 16,
            "car_number": 12,
            "driver_display": "Reviewed DRIVER",
            "constructor_display": "Known Team",
            "q1": "1:16.525",
            "q2": None,
            "q3": None,
        }
    ],
}


def test_exact_review_preserves_literal_values_without_car_number_identity_join():
    rows = reviewed_qualifying(TEXT, EVENT, DRIVERS, CONSTRUCTORS, REVIEW, "exact_pdf").to_pylist()
    assert rows[1]["driver_id"] == "reviewed"
    assert rows[1]["q1_seconds"] == 76.525
    assert rows[1]["q2_seconds"] is None


@pytest.mark.parametrize(
    "field,value", [("q1", "1:16.526"), ("position", 15), ("car_number", 13), ("q2", "1:15.000")]
)
def test_changed_transcription_cannot_synthesize_unprinted_values(field, value):
    altered = deepcopy(REVIEW)
    altered["rows"][0][field] = value
    with pytest.raises(ValueError, match="contradicts"):
        reviewed_qualifying(TEXT, EVENT, DRIVERS, CONSTRUCTORS, altered, "exact_pdf")


def test_transcription_cannot_apply_to_another_pdf_or_ambiguous_row():
    with pytest.raises(ValueError, match="exact PDF"):
        reviewed_qualifying(TEXT, EVENT, DRIVERS, CONSTRUCTORS, REVIEW, "changed_pdf")
    with pytest.raises(ValueError, match="ambiguous"):
        reviewed_qualifying(
            TEXT + "\ndamaged name", EVENT, DRIVERS, CONSTRUCTORS, REVIEW, "exact_pdf"
        )
