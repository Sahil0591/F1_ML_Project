from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from f1_ml_predictor.features.contracts import PublishedTable
from f1_ml_predictor.time import require_known_by
from f1_ml_predictor.trust.evidence import AvailabilityEvidence, EvidenceClass, table_hash
from f1_ml_predictor.trust.fia import (
    FiaDocumentMetadata,
    FiaDocumentStatus,
    fia_publication_evidence,
)


def test_direct_minute_precision_uses_upper_availability_bound():
    table = pa.table({"value": [1]})
    published = datetime(2025, 3, 15, 6, 30, tzinfo=UTC)
    metadata = FiaDocumentMetadata(
        "https://www.fia.com/system/files/q.pdf",
        "23",
        "a" * 64,
        "2025-03-15 07:30",
        "CET",
        published,
        FiaDocumentStatus.PROVISIONAL,
        table_hash(table),
        True,
        "exact registry/document/value audit",
        60,
    )
    proof = fia_publication_evidence(metadata, table)
    assert proof.source_published_at == published
    assert proof.available_at == published + timedelta(minutes=1)
    assert proof.to_dict()["publication_precision_seconds"] == 60
    assert PublishedTable(table, proof.available_at, proof.reference, proof).tier.value == "Gold"
    with pytest.raises(ValueError):
        require_known_by(proof.available_at, published + timedelta(seconds=30))
    require_known_by(proof.available_at, published + timedelta(minutes=2))


def test_precision_cannot_hide_arbitrary_backdating_or_upgrade_current_state():
    published = datetime(2025, 3, 15, 6, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="precision"):
        AvailabilityEvidence(
            EvidenceClass.CURRENT_STATE_ONLY, "api now", publication_precision_seconds=60
        )
    with pytest.raises(ValueError, match="precision"):
        AvailabilityEvidence(
            EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP,
            "doc",
            published + timedelta(hours=1),
            artifact_sha256="a" * 64,
            source_published_at=published,
            audited=True,
            publication_precision_seconds=60,
        )
