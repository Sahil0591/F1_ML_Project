import json
from dataclasses import replace

import httpx
import pyarrow.parquet as pq
import pytest
from test_trust_features import CUTOFF, certified_pub, inputs, qualifying_rows, roster_rows

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.trust import historical
from f1_ml_predictor.trust.evidence import AvailabilityEvidence, EvidenceClass, table_hash


def core_request(tmp_path, monkeypatch, *, kind=EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP):
    original = inputs(certified=True)
    event = original.event
    proof = AvailabilityEvidence(
        kind,
        "event-proof",
        event.available_at,
        artifact_sha256=table_hash(event.as_table()),
        source_published_at=event.available_at
        if kind == EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP
        else None,
        archive_version="v1" if kind == EvidenceClass.VERSIONED_ARCHIVE else None,
        reconstruction_method="bounded"
        if kind == EvidenceClass.CONSERVATIVE_RECONSTRUCTION
        else None,
        audited=True,
    )
    value = replace(
        original,
        event=replace(event, evidence=proof),
        rosters=(certified_pub(roster_rows(), CUTOFF, "roster-proof", kind=kind, audited=True),),
        qualifying=(
            certified_pub(qualifying_rows(), CUTOFF, "qualifying-proof", kind=kind, audited=True),
        ),
    )
    pdf = tmp_path / "qualification.pdf"
    pdf.write_bytes(b"%PDF fixture exact version")
    registry = tmp_path / "registry.html"
    registry.write_bytes(b'<a href="https://www.fia.com/system/files/fixture.pdf">Doc 23</a>')
    request = {
        "document_bindings": [
            {
                "path": pdf.name,
                "sha256": file_sha256(pdf),
                "registry_path": registry.name,
                "registry_sha256": file_sha256(registry),
                "document_id": "23",
                "status": "provisional",
                "version_audited": True,
                "audit_reference": "fixture audit",
                "event_id": value.event.event.partition(),
                "document_url": "https://www.fia.com/system/files/fixture.pdf",
                "latest_at_cutoff_audited": True,
                "table_bindings": [
                    {
                        "reference": e.reference,
                        "table_sha256": e.artifact_sha256,
                        "available_at_utc": e.available_at.isoformat(),
                    }
                    for e in [
                        value.event.evidence,
                        value.rosters[0].evidence,
                        value.qualifying[0].evidence,
                    ]
                ],
            }
        ]
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(historical, "load_feature_request", lambda *_: (value, CUTOFF))
    monkeypatch.setattr(historical, "inspect_pdf", lambda _: {"document_id": "23"})
    return request_path, request


def test_lean_core_keeps_missing_optional_features_and_freezes_manifest(tmp_path, monkeypatch):
    path, _ = core_request(tmp_path, monkeypatch)
    result = historical.reconstruct_gold_core(path, tmp_path)
    table = pq.ParquetFile(tmp_path / result["features"]["path"]).read()
    assert table["forecast_temperature_2m"].null_count == 2
    assert table["practice_best_seconds"].null_count == 2
    assert set(table["benchmark_tier"].to_pylist()) == {"Gold"}
    manifest = tmp_path / result["evidence_manifest"]
    assert json.loads(manifest.read_text())["schema_version"] == "gold-core-v1"
    original = manifest.read_bytes()
    assert historical.reconstruct_gold_core(path, tmp_path)["snapshot_id"] == result["snapshot_id"]
    assert manifest.read_bytes() == original
    assert result["selection_eligible"] is False
    assert result["benchmark"]["excluded_races"] == 1


@pytest.mark.parametrize(
    "kind", [EvidenceClass.CURRENT_STATE_ONLY, EvidenceClass.CONSERVATIVE_RECONSTRUCTION]
)
def test_core_never_promotes_current_or_reconstructed_evidence(tmp_path, monkeypatch, kind):
    path, _ = core_request(tmp_path, monkeypatch, kind=kind)
    with pytest.raises(ValueError, match="direct Gold"):
        historical.reconstruct_gold_core(path, tmp_path)


@pytest.mark.parametrize("change", ["hash", "document_id", "recalled", "unaudited", "unbound"])
def test_historical_core_rejects_document_version_failures(tmp_path, monkeypatch, change):
    path, request = core_request(tmp_path, monkeypatch)
    binding = request["document_bindings"][0]
    if change == "hash":
        binding["sha256"] = "0" * 64
    elif change == "document_id":
        binding["document_id"] = "24"
    elif change == "recalled":
        binding["status"] = "recalled"
    elif change == "unaudited":
        binding["version_audited"] = False
    else:
        binding["table_bindings"] = []
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        historical.reconstruct_gold_core(path, tmp_path)
    assert not (tmp_path / "data/benchmarks/gold_core_registry.json").exists()


def test_discovery_reports_coverage_without_certifying_current_downloads(tmp_path, monkeypatch):
    url = "https://www.fia.com/system/files/q.pdf"
    catalog = tmp_path / "candidates.json"
    catalog.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "season": 2025,
                        "round": 12,
                        "index_url": "https://www.fia.com/documents/test",
                        "qualifying_url": url,
                        "document_id": "23",
                    }
                ]
            }
        )
    )

    def respond(request):
        content = (
            b"%PDF fixture"
            if request.url.path.endswith(".pdf")
            else (b'<a href="/system/files/q.pdf">Provisional Qualifying Classification</a>')
        )
        return httpx.Response(200, content=content)

    monkeypatch.setattr(historical, "inspect_pdf", lambda _: {"document_id": "23"})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        report = historical.discover_auditability(catalog, tmp_path, http_client=client)
    row = report["candidates"][0]
    assert row["expected_gold_core_completeness"] == 3
    assert row["eligible_gold"] is False
    assert report["status"] == "insufficient_data"
    assert row["qualifying_artifact"]["evidence_classification"] == "current_state_only"
    assert "publication_clock_not_audited" in row["exclusion_reasons"]
    object_path = tmp_path / row["qualifying_artifact"]["path"]
    assert file_sha256(object_path) == row["qualifying_artifact"]["sha256"]


def test_discovery_does_not_hide_replaced_qualifying_document(tmp_path, monkeypatch):
    url = "https://www.fia.com/system/files/q.pdf"
    catalog = tmp_path / "candidates.json"
    catalog.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "season": 2025,
                        "round": 12,
                        "index_url": "https://www.fia.com/documents/test",
                        "qualifying_url": url,
                        "document_id": "23",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(historical, "inspect_pdf", lambda _: {"document_id": "31"})
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b"%PDF fixture"
                if request.url.path.endswith(".pdf")
                else b'<a href="/system/files/q.pdf">Qualifying</a>',
            )
        )
    ) as client:
        row = historical.discover_auditability(catalog, tmp_path, http_client=client)["candidates"][
            0
        ]
    assert "document_number_changed" in row["exclusion_reasons"]
    assert row["expected_gold_core_completeness"] == 0


def test_immutable_audit_artifacts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "proof.json"
    historical._immutable(path, b"first")
    historical._immutable(path, b"first")
    with pytest.raises(ValueError, match="immutable"):
        historical._immutable(path, b"revised")
    assert path.read_bytes() == b"first"
