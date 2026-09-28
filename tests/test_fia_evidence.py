from dataclasses import replace
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from f1_ml_predictor.trust.evidence import BenchmarkTier, EvidenceClass, table_hash
from f1_ml_predictor.trust.fia import (
    FiaDocumentMetadata,
    FiaDocumentStatus,
    fia_publication_evidence,
)


def metadata(table: pa.Table, **changes: object) -> FiaDocumentMetadata:
    values: dict[str, object] = {
        "url": "https://www.fia.com/sites/default/files/final_classification.pdf",
        "document_id": "2025-01-document-42-revision-2",
        "document_sha256": "a" * 64,
        "printed_publication_timestamp": "2025-03-16 18:30",
        "printed_timezone": "CET",
        "published_at": datetime(2025, 3, 16, 17, 30, tzinfo=UTC),
        "status": FiaDocumentStatus.FINAL,
        "table_sha256": table_hash(table),
        "audited": True,
        "audit_reference": "audit:42-r2",
    }
    values.update(changes)
    return FiaDocumentMetadata(**values)  # type: ignore[arg-type]


def test_offline_explicit_fia_publication_evidence() -> None:
    table = pa.table({"position": [1, 2]})
    record = metadata(table)
    evidence = fia_publication_evidence(record, table)
    assert evidence.kind == EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP
    assert evidence.available_at == record.published_at
    assert evidence.tier == BenchmarkTier.GOLD
    assert record.document_sha256 in evidence.reference
    assert record.document_id in evidence.reference
    evidence.validate_table(table, record.published_at)


def test_official_url_does_not_automatically_certify_audit() -> None:
    table = pa.table({"position": [1]})
    evidence = fia_publication_evidence(metadata(table, audited=False, audit_reference=None), table)
    assert evidence.audited is False
    assert evidence.tier == BenchmarkTier.DEVELOPMENT


@pytest.mark.parametrize("status", list(FiaDocumentStatus))
def test_revision_and_recall_are_explicit(status: FiaDocumentStatus) -> None:
    table = pa.table({"position": [1]})
    record = metadata(table, status=status)
    if status == FiaDocumentStatus.RECALLED:
        with pytest.raises(ValueError, match="recalled"):
            fia_publication_evidence(record, table)
    else:
        assert status.value in fia_publication_evidence(record, table).reference


def test_exact_table_version_is_required() -> None:
    table = pa.table({"position": [1]})
    with pytest.raises(ValueError, match="exact"):
        fia_publication_evidence(metadata(table), pa.table({"position": [2]}))


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://www.fia.com/doc.pdf"},
        {"url": "https://fia.com.example.org/doc.pdf"},
        {"url": "https://user@www.fia.com/doc.pdf"},
        {"document_id": ""},
        {"document_sha256": ""},
        {"printed_timezone": "local"},
        {"printed_publication_timestamp": "race end"},
        {"published_at": datetime(2025, 3, 16, 18, 30, tzinfo=UTC)},
        {"published_at": datetime(2025, 3, 16, 17, 30)},
        {"audit_reference": None},
        {"status": "final"},
    ],
)
def test_invalid_or_inferred_publication_metadata_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        metadata(pa.table({"position": [1]}), **changes)


def test_printed_cest_is_converted_explicitly() -> None:
    table = pa.table({"position": [1]})
    record = replace(
        metadata(table),
        printed_timezone="CEST",
        published_at=datetime(2025, 3, 16, 16, 30, tzinfo=UTC),
    )
    assert fia_publication_evidence(record, table).available_at.hour == 16  # type: ignore[union-attr]
