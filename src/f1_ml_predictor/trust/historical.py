"""Bounded FIA discovery and exact-version Gold Core reconstruction.

Discovery estimates auditability. It never certifies a present-day download.
Reconstruction consumes explicit audited feature requests from the existing
feature contract, plus retained document bytes and separate outcome provenance.
"""

import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import (
    _safe_file,
    _validate_feature_table,
    build_benchmarks,
    file_sha256,
)
from f1_ml_predictor.features.manifest import load_feature_request
from f1_ml_predictor.features.snapshot import NUMERIC_FEATURES, build_snapshot
from f1_ml_predictor.features.storage import persist_snapshot
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.time import require_known_by
from f1_ml_predictor.trust.evidence import BenchmarkTier, EvidenceClass
from f1_ml_predictor.trust.outcomes import validate_audited_outcomes

CORE_SCHEMA_VERSION = "gold-core-v1"
CORE_FEATURES = frozenset(
    {
        "qualifying_position",
        "qualifying_last_session_seconds",
        "grid_position",
        "teammate_qualifying_position_delta",
        "recent_finish_mean",
        "recent_dnf_rate",
        "constructor_recent_finish_mean",
        "driver_championship_points",
        "constructor_championship_points",
        "circuit_length_km",
        "is_street_circuit",
        "forecast_temperature_2m",
        "forecast_precipitation_probability",
        "forecast_wind_speed_10m",
    }
)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        with pending.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Creating the final hard link is atomic and cannot replace a file.
            os.link(pending, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(
                    "immutable audit artifact already contains different bytes"
                ) from None
    finally:
        pending.unlink(missing_ok=True)


class _DocumentLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        href = attributes.get("href")
        if tag == "a" and href and ".pdf" in href.lower():
            self.current = {"url": urljoin("https://www.fia.com", href), "title": ""}

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current["title"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current is not None:
            self.current["title"] = " ".join(self.current["title"].split())
            self.links.append(self.current)
            self.current = None


def _official(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"www.fia.com", "fia.com"}
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        raise ValueError("historical discovery requires an official FIA HTTPS URL")


def _retain_response(root: Path, response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    if len(response.content) > 20 * 1024 * 1024:
        raise ValueError("FIA audit response exceeds the bounded artifact size")
    digest = hashlib.sha256(response.content).hexdigest()
    extension = ".pdf" if response.content.startswith(b"%PDF") else ".html"
    path = root / "data/raw/fia_audit/objects" / f"{digest}{extension}"
    _immutable(path, response.content)
    metadata = {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "url": str(response.url),
        "provider": "Formula1" if response.url.host == "www.formula1.com" else "FIA",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "request_parameters": dict(response.request.url.params),
        "provider_version": response.headers.get("etag"),
        "last_modified": response.headers.get("last-modified"),
        "content_type": response.headers.get("content-type"),
        "evidence_classification": EvidenceClass.CURRENT_STATE_ONLY.value,
    }
    record_hash = hashlib.sha256(_json(metadata)).hexdigest()
    _immutable(root / "data/raw/fia_audit/requests" / f"{record_hash}.json", _json(metadata))
    return metadata


def discover_auditability(
    candidates_path: Path,
    root: Path,
    *,
    limit: int = 17,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Rank a bounded researched pool using live official document-link coverage.

    Missing or ambiguous clocks and recalled bytes remain explicit audit tasks.
    Schedule/identity publication proof must be supplied before reconstruction.
    """
    if isinstance(limit, bool) or not 1 <= limit <= 20:
        raise ValueError("initial candidate pool must contain between one and twenty races")
    root = root.resolve()
    catalog = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = catalog["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("candidate catalog requires a candidates list")
    own_client = http_client is None
    client = http_client or httpx.Client(
        timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
    )
    records = []
    try:
        for item in candidates[:limit]:
            record = dict(item)
            record.update({"status": "audit_required", "eligible_gold": False})
            try:
                url = item["index_url"]
                _official(url)
                response = client.get(url)
                record["registry_artifact"] = _retain_response(root, response)
                parser = _DocumentLinks()
                parser.feed(response.text)
                links = list({link["url"]: link for link in parser.links}.values())
                expected = item["qualifying_url"]
                qualification = next((link for link in links if link["url"] == expected), None)
                record["document_links"] = links
                record["expected_gold_core_completeness"] = 0
                reasons = [
                    "schedule_publication_not_audited",
                    "publication_clock_not_audited",
                    "exact_document_version_not_audited",
                    "final_labels_not_audited",
                ]
                if qualification is None:
                    reasons.append("expected_qualifying_version_not_in_registry")
                else:
                    _official(expected)
                    if own_client:
                        time.sleep(1)
                    pdf_response = client.get(expected)
                    record["qualifying_artifact"] = _retain_response(root, pdf_response)
                    if not pdf_response.content.startswith(b"%PDF"):
                        raise ValueError("qualifying response is not a PDF")
                    record["expected_gold_core_completeness"] = 3
                    record["document_identity"] = inspect_pdf(
                        root / record["qualifying_artifact"]["path"]
                    )
                    if record["document_identity"]["document_id"] != str(item["document_id"]):
                        reasons.append("document_number_changed")
                        record["expected_gold_core_completeness"] = 0
                record["exclusion_reasons"] = reasons
            except (ValueError, OSError, KeyError, httpx.HTTPError) as exc:
                record.update(
                    {
                        "status": "discovery_failed",
                        "exclusion_reasons": [str(exc)],
                        "expected_gold_core_completeness": 0,
                    }
                )
            records.append(record)
            if own_client:
                time.sleep(1)
    finally:
        if own_client:
            client.close()
    records.sort(
        key=lambda row: (
            -row["expected_gold_core_completeness"],
            -row.get("season", 0),
            -row.get("round", 0),
        )
    )
    report = {
        "version": 1,
        "schema_version": CORE_SCHEMA_VERSION,
        "minimum_gold_races": 8,
        "eligible_gold_races": 0,
        "status": "insufficient_data",
        "candidates": records,
        "catalog_sha256": file_sha256(candidates_path),
    }
    digest = hashlib.sha256(_json(report)).hexdigest()
    _immutable(
        root / "data/benchmarks/historical_audit" / f"auditability-{digest}.json", _json(report)
    )
    return report


def inspect_pdf(path: Path) -> dict[str, Any]:
    """Extract retained bytes for review; PDF issue time is never publication proof."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("FIA PDF inspection requires the optional audit dependency") from exc
    reader = PdfReader(path)
    text = "\n".join(page.extract_text(extraction_mode="layout") for page in reader.pages)
    cover = reader.pages[0].extract_text() or ""
    # Layout extraction can omit a rotated cover. Its document number still
    # comes from the PDF, using ordinary extraction of that same first page.
    pattern = r"\bDoc(?:ument|\s*No\.?)?\s*:?\s*(\d+)\b"
    match = re.search(pattern, cover, re.IGNORECASE) or re.search(pattern, text, re.IGNORECASE)
    return {
        "document_id": match[1] if match else None,
        "pages": len(reader.pages),
        "text": text,
        "cover_text": cover,
        "document_sha256": file_sha256(path),
        "extraction_version": "pypdf-layout-cover-v2",
    }


def reconstruct_gold_core(request_path: Path, root: Path) -> dict[str, Any]:
    """Audit a lean feature request, freeze its evidence, and update a Gold registry.

    This accepts direct audited publication/archive proof, never promotes
    reconstruction or a current API download. Outcome proof is kept separate.
    """
    root = root.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    inputs, cutoff = load_feature_request(request_path, root)
    if inputs.sessions:
        raise ValueError("Gold Core omits practice and tyre inputs; use a Full request")
    publications = [
        *inputs.rosters,
        *inputs.qualifying,
        *(item.publication for item in inputs.history),
        *inputs.forecasts,
    ]
    publications.extend(value for value in (inputs.standings, inputs.circuit) if value is not None)
    evidence = [inputs.event.evidence, *(value.evidence for value in publications)]
    bindings = request.get("document_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("historical Gold requires retained exact publication document bindings")
    bound_proofs: dict[str, list[dict[str, Any]]] = {}
    for binding in bindings:
        artifact = _safe_file(root, binding["path"], binding["sha256"])
        publication_registry = _safe_file(
            root, binding["registry_path"], binding["registry_sha256"]
        )
        inspected = inspect_pdf(artifact)
        if inspected["document_id"] != str(binding["document_id"]):
            raise ValueError("retained PDF document identity contradicts its audited version")
        if binding.get("status") == "recalled":
            raise ValueError("recalled publication cannot certify a historical feature version")
        if binding.get("version_audited") is not True or not binding.get("audit_reference"):
            raise ValueError("publication version needs an explicit completed audit")
        if binding.get("latest_at_cutoff_audited") is not True:
            raise ValueError("latest required-field state at cutoff must be audited")
        if binding.get("event_id") != inputs.event.event.partition():
            raise ValueError("document binding belongs to another event")
        _official(binding["document_url"])
        # Audit reference includes the exact official registry, not an API replay.
        parser = _DocumentLinks()
        parser.feed(publication_registry.read_text(encoding="utf-8"))
        if binding["document_url"] not in {link["url"] for link in parser.links}:
            raise ValueError("exact document URL is absent from retained publication registry")
        for bound in binding["table_bindings"]:
            bound_proofs.setdefault(bound["reference"], []).append(bound)
        for supporting in binding.get("supporting_artifacts", []):
            _safe_file(root, supporting["path"], supporting["sha256"])
    for proof in evidence:
        if proof is None or proof.tier != BenchmarkTier.GOLD or not proof.audited:
            raise ValueError(
                "historical Core requires audited direct Gold evidence for every input"
            )
        if proof.kind not in {
            EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP,
            EvidenceClass.VERSIONED_ARCHIVE,
        }:
            raise ValueError("historical Core cannot relabel current or reconstructed inputs")
        assert proof.available_at is not None
        matching = [
            bound
            for bound in bound_proofs.get(proof.reference, [])
            if bound.get("table_sha256") == proof.artifact_sha256
            and bound.get("available_at_utc") == proof.available_at.isoformat()
        ]
        if len(matching) != 1:
            raise ValueError("feature evidence lacks its retained exact-version document binding")
        assert proof.available_at is not None
        require_known_by(proof.available_at, cutoff)
    table = build_snapshot(inputs, cutoff, certified_only=True)
    rows, tier = _validate_feature_table(table, inputs.event.event, cutoff, "post_qualifying")
    if tier != BenchmarkTier.GOLD:
        raise ValueError("Core snapshot failed the unchanged Gold policy")
    for row in rows:
        if any(row[name] is not None for name in set(NUMERIC_FEATURES) - CORE_FEATURES):
            raise ValueError("Gold Core request contains unsupported richer features")
    feature_path = persist_snapshot(StoragePaths(root), table)
    persisted = pq.ParquetFile(feature_path).read()
    if not persisted.equals(table, check_metadata=False):
        raise ValueError("persisted snapshot contents differ from the audited feature table")
    matrix = [
        {
            "driver_id": row["driver_id"],
            "feature": name,
            "missing": row[name] is None,
            "available_at": row["feature_timestamp"].isoformat() if row[name] is not None else None,
            "evidence": json.loads(row["feature_evidence"])[name],
        }
        for row in rows
        for name in NUMERIC_FEATURES
    ]
    snapshot_id = file_sha256(feature_path)
    manifest = {
        "version": 1,
        "schema_version": CORE_SCHEMA_VERSION,
        "normalization_version": "feature-v2",
        "event_id": inputs.event.event.partition(),
        "snapshot_id": snapshot_id,
        "prediction_cutoff_utc": cutoff.isoformat(),
        "cutoff_kind": "post_qualifying",
        "benchmark_tier": "Gold",
        "request_sha256": hashlib.sha256(
            _json(
                {
                    key: value
                    for key, value in request.items()
                    if key not in {"outcomes", "outcome_document_bindings"}
                }
            )
        ).hexdigest(),
        "document_bindings": bindings,
        "feature_manifest": matrix,
    }
    evidence_dir = root / "data/features/historical_evidence" / snapshot_id
    _immutable(evidence_dir / "manifest.json", _json(manifest))
    _immutable(
        evidence_dir / "request.json",
        _json(
            {
                key: value
                for key, value in request.items()
                if key not in {"outcomes", "outcome_document_bindings"}
            }
        ),
    )
    _immutable(evidence_dir / "availability.json", _json(matrix))
    item: dict[str, Any] = {
        "event_id": inputs.event.event.partition(),
        "prediction_timestamp": cutoff.isoformat(),
        "cutoff_kind": "post_qualifying",
        "schema_version": CORE_SCHEMA_VERSION,
        "features": {"path": feature_path.relative_to(root).as_posix(), "sha256": snapshot_id},
        "outcomes": request.get("outcomes"),
    }
    registry_path = root / "data/benchmarks/gold_core_registry.json"
    registry: dict[str, Any] = (
        json.loads(registry_path.read_text())
        if registry_path.exists()
        else {"version": 1, "races": []}
    )
    if item["outcomes"] is not None:
        outcome = item["outcomes"]
        outcome_path = _safe_file(root, outcome["path"], outcome["sha256"])
        outcome_table = pq.ParquetFile(outcome_path).read()
        validate_audited_outcomes(
            outcome_table, field_roster={(inputs.event.event, row["driver_id"]) for row in rows}
        )
        target_bindings = request.get("outcome_document_bindings")
        if not isinstance(target_bindings, list) or not target_bindings:
            raise ValueError("audited labels require separate exact final document bindings")
        for target in target_bindings:
            document = _safe_file(root, target["path"], target["sha256"])
            if (
                target.get("event_id") != item["event_id"]
                or target.get("status") != "final"
                or target.get("version_audited") is not True
                or target.get("latest_final_audited") is not True
                or not target.get("audit_reference")
                or target.get("outcome_sha256") != outcome["sha256"]
                or inspect_pdf(document)["document_id"] != str(target["document_id"])
            ):
                raise ValueError("final outcome document binding contradicts audited labels")
            _official(target["document_url"])
            target_registry = _safe_file(root, target["registry_path"], target["registry_sha256"])
            parser = _DocumentLinks()
            parser.feed(target_registry.read_text(encoding="utf-8"))
            if target["document_url"] not in {link["url"] for link in parser.links}:
                raise ValueError("final outcome URL is absent from retained publication registry")
            if any(
                row["audit_reference"] != target["audit_reference"]
                or row["label_available_at"].isoformat() != target.get("label_available_at_utc")
                for row in outcome_table.to_pylist()
            ):
                raise ValueError("target audit and availability are not bound to the outcome rows")
        if any(row["label_available_at"] <= cutoff for row in outcome_table.to_pylist()):
            raise ValueError("target publication must follow the predictive cutoff")
        _immutable(
            evidence_dir / "targets" / f"{outcome['sha256']}.json",
            _json(
                {
                    "snapshot_id": snapshot_id,
                    "outcomes": outcome,
                    "document_bindings": target_bindings,
                }
            ),
        )
    cohort = (item["event_id"], item["prediction_timestamp"])
    previous = next(
        (
            race
            for race in registry["races"]
            if (race["event_id"], race["prediction_timestamp"]) == cohort
        ),
        None,
    )
    if previous is not None and previous != item:
        previous_predictive = {key: value for key, value in previous.items() if key != "outcomes"}
        current_predictive = {key: value for key, value in item.items() if key != "outcomes"}
        if previous_predictive != current_predictive or previous.get("outcomes") is not None:
            raise ValueError(
                "registered immutable Core snapshot or attached target cannot be replaced"
            )
        previous["outcomes"] = item["outcomes"]
    if previous is None:
        registry["races"].append(item)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    # Registry is an index; source bundles and evidence remain immutable.
    temporary = registry_path.with_suffix(".pending")
    temporary.write_bytes(_json(registry))
    temporary.replace(registry_path)
    report = build_benchmarks(root, root / "data/benchmarks/gold_core", registry_path)
    return {
        "snapshot_id": snapshot_id,
        "features": item["features"],
        "evidence_manifest": (evidence_dir / "manifest.json").relative_to(root).as_posix(),
        "benchmark": report,
        "selection_eligible": len(report["datasets"]["Gold"]["events"]) >= 8,
    }


def withdraw_gold_core(root: Path, snapshot_id: str, reason: str) -> dict[str, Any]:
    """Withdraw an invalid certification from the index, preserving frozen bytes."""
    root = root.resolve()
    if not reason.strip():
        raise ValueError("withdrawal requires an explicit audit reason")
    path = root / "data/benchmarks/gold_core_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    matches = [race for race in registry["races"] if race["features"]["sha256"] == snapshot_id]
    if len(matches) != 1:
        raise ValueError("withdrawal requires one exact registered snapshot hash")
    record = {
        "snapshot": matches[0],
        "reason": reason,
        "withdrawn_at_utc": datetime.now(UTC).isoformat(),
    }
    registry.setdefault("withdrawn_races", []).append(record)
    registry["races"] = [race for race in registry["races"] if race != matches[0]]
    temporary = path.with_suffix(".pending")
    temporary.write_bytes(_json(registry))
    temporary.replace(path)
    report = build_benchmarks(root, root / "data/benchmarks/gold_core", path)
    return {"withdrawal": record, "benchmark": report}
