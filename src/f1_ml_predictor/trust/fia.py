"""Offline evidence for exact FIA documents and explicitly audited tables."""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from urllib.parse import urlsplit

import pyarrow as pa

from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.evidence import AvailabilityEvidence, EvidenceClass, table_hash


class FiaDocumentStatus(StrEnum):
    PROVISIONAL = "provisional"
    FINAL = "final"
    REVISED = "revised"
    RECALLED = "recalled"


@dataclass(frozen=True, slots=True)
class FiaDocumentMetadata:
    url: str
    document_id: str
    document_sha256: str
    printed_publication_timestamp: str
    printed_timezone: str
    published_at: datetime
    status: FiaDocumentStatus
    table_sha256: str
    audited: bool = False
    audit_reference: str | None = None
    publication_precision_seconds: int = 0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in {"fia.com", "www.fia.com"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or not parsed.path.strip("/")
        ):
            raise ValueError("FIA evidence requires an official HTTPS document URL")
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("exact FIA document identity is required")
        for value in (self.document_sha256, self.table_sha256):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("exact document and table SHA-256 hashes are required")
        if not isinstance(self.status, FiaDocumentStatus):
            raise ValueError("explicit FIA document status is required")
        if not isinstance(self.audited, bool):
            raise ValueError("explicit audit flag is required")
        if type(
            self.publication_precision_seconds
        ) is not int or self.publication_precision_seconds not in {0, 60}:
            raise ValueError("publication precision must be exact or one minute")
        if self.audited and (not self.audit_reference or not self.audit_reference.strip()):
            raise ValueError("audited FIA metadata requires an audit reference")
        require_utc(self.published_at, "published_at")
        if self.printed_timezone not in {"CET", "CEST", "UTC"}:
            raise ValueError("printed publication timezone must be explicit CET, CEST or UTC")
        local: datetime | None = None
        for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M"):
            try:
                local = datetime.strptime(self.printed_publication_timestamp, pattern)
                break
            except ValueError:
                continue
        if local is None:
            raise ValueError("unsupported printed publication timestamp")
        offset = {"UTC": 0, "CET": 1, "CEST": 2}[self.printed_timezone]
        expected = local.replace(tzinfo=timezone(timedelta(hours=offset))).astimezone(UTC)
        if expected != self.published_at:
            raise ValueError("parsed UTC publication time contradicts printed timestamp/timezone")


def fia_publication_evidence(
    metadata: FiaDocumentMetadata, table: pa.Table
) -> AvailabilityEvidence:
    """Bind supplied publication metadata to an exact table; no scraping or time inference."""
    if metadata.status == FiaDocumentStatus.RECALLED:
        raise ValueError("recalled FIA documents cannot certify availability")
    if metadata.table_sha256 != table_hash(table):
        raise ValueError("FIA metadata does not bind this exact audited table")
    reference = json.dumps(
        {
            "url": metadata.url,
            "document_id": metadata.document_id,
            "document_sha256": metadata.document_sha256,
            "printed_publication_timestamp": metadata.printed_publication_timestamp,
            "printed_timezone": metadata.printed_timezone,
            "status": metadata.status.value,
            "audit_reference": metadata.audit_reference,
            "publication_precision_seconds": metadata.publication_precision_seconds,
        },
        sort_keys=True,
    )
    return AvailabilityEvidence(
        kind=EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP,
        reference=reference,
        available_at=metadata.published_at
        + timedelta(seconds=metadata.publication_precision_seconds),
        artifact_sha256=metadata.table_sha256,
        source_published_at=metadata.published_at,
        audited=metadata.audited,
        publication_precision_seconds=metadata.publication_precision_seconds,
    )
