import json
from datetime import UTC, datetime

import pytest

from f1_ml_predictor.trust.winter import (
    _bind_pdf_identity,
    _publication,
    _resume_request,
    constructor_aliases_for_season,
    latest_final_record,
    qualifying_at_cutoff,
    registry_rows,
    roster_at_cutoff,
)

PREFIX = "https://www.fia.com/system/files/decision-document/2025_test_grand_prix_-_"


def record(identifier, title, clock, *, recalled=False):
    return {
        "document_id": str(identifier),
        "title": title,
        "publication_cet": f"08.03.25 {clock}",
        "url": PREFIX + f"{identifier}.pdf",
        "recalled": recalled,
    }


def test_main_qualifying_state_ignores_sprint_but_not_earlier_revisions():
    main = record(30, "Provisional Qualifying Classification", "16:30")
    sprint = record(20, "Final Sprint Qualifying Classification", "10:00")
    later = record(31, "Final Qualifying Classification", "18:00")
    cutoff = datetime(2025, 3, 8, 15, 32, tzinfo=UTC)
    qualifying_at_cutoff([sprint, main, later], main, cutoff)
    with pytest.raises(ValueError, match="another version"):
        qualifying_at_cutoff([main, record(31, main["title"], "16:32")], main, cutoff)
    with pytest.raises(ValueError, match="another version"):
        qualifying_at_cutoff([main], main, datetime(2025, 3, 8, 15, 30, tzinfo=UTC))


def test_summer_cet_label_uses_later_possible_utc_bound():
    summer = record(30, "Provisional Qualifying Classification", "16:30")
    summer["publication_cet"] = "05.07.25 16:30"
    assert _publication(summer) == datetime(2025, 7, 5, 15, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="another version"):
        qualifying_at_cutoff([summer], summer, datetime(2025, 7, 5, 15, 30, tzinfo=UTC))
    qualifying_at_cutoff([summer], summer, datetime(2025, 7, 5, 15, 31, tzinfo=UTC))


def test_latest_roster_replacement_covers_old_recall_but_uncertain_release_blocks():
    old = record(11, "Entry List", "10:00", recalled=True)
    selected = record(13, "Entry List V2", "11:00")
    cutoff = datetime(2025, 3, 8, 15, 32, tzinfo=UTC)
    roster_at_cutoff([old, selected], selected, cutoff, from_qualifying=False)
    newer = record(14, "Entry List V3", "16:32")
    with pytest.raises(ValueError, match="latest"):
        roster_at_cutoff([old, selected, newer], selected, cutoff, from_qualifying=False)
    main = record(30, "Provisional Qualifying Classification", "16:30")
    with pytest.raises(ValueError, match="review"):
        roster_at_cutoff([newer], main, cutoff, from_qualifying=True)


def test_final_label_requires_latest_version_and_no_unreviewed_later_rulings():
    final = record(50, "Final Race Classification", "20:30")
    points = record(51, "Championship Points", "20:35")
    assert latest_final_record([final, points], final["url"], "50") == final
    ruling = record(52, "Decision - Car 4 - Technical Non-Compliance", "20:40")
    with pytest.raises(ValueError, match="review"):
        latest_final_record([final, ruling], final["url"], "50")
    same_minute_ruling = record(52, ruling["title"], "20:30")
    with pytest.raises(ValueError, match="review"):
        latest_final_record([final, same_minute_ruling], final["url"], "50")
    revision = record(53, "Final Race Classification V2", "21:00")
    with pytest.raises(ValueError, match="latest"):
        latest_final_record([final, revision], final["url"], "50")
    assert latest_final_record([final, ruling, revision], revision["url"], "53") == revision


def test_later_review_requires_exact_registry_identity(tmp_path):
    final = record(50, "Final Race Classification", "20:30")
    ruling = record(52, "Decision - Williams Petition for Right of Review", "20:40")
    review = {
        "later_documents": [{"document_id": "52", "title": ruling["title"], "url": ruling["url"]}],
        "conclusion": "classification_cannot_be_amended",
    }
    changed = dict(ruling, document_id="53")
    with pytest.raises(ValueError, match="differ"):
        latest_final_record([final, changed], final["url"], "50", review=review, root=tmp_path)


def test_recalled_rows_remain_visible_and_cannot_be_selected():
    html = '<li class="document-row"><span>Doc 50 - Final Race Classification</span>'
    html += '<a href="/system/files/decision-document/50.pdf">PDF</a>'
    html += "<span>Published on 08.03.25 20:30 CET</span><span>RECALLED</span></li>"
    rows = registry_rows(html)
    assert rows[0]["recalled"] is True
    with pytest.raises(ValueError, match="recalled"):
        latest_final_record(rows, rows[0]["url"], "50")


def test_constructor_variants_are_scoped_to_the_document_season():
    aliases = constructor_aliases_for_season(2024)
    assert aliases["RB Honda RBPT"] == "rb"
    assert aliases["Kick Sauber Ferrari"] == "sauber"
    assert "RB Honda RBPT" not in constructor_aliases_for_season(2025)
    assert "Kick Sauber Ferrari" in constructor_aliases_for_season(2025)
    assert "Kick Sauber Ferrari" not in constructor_aliases_for_season(2026)


def test_legacy_registry_preserves_publication_without_inventing_document_number():
    html = (
        '<li class="document-row key-1668870865"><a href="/sites/default/files/'
        'decision-document/2022 Abu Dhabi Grand Prix - Provisional Qualifying Classification.pdf">'
        '<div class="title">Provisional Qualifying Classification</div>'
        '<div class="published">Published on <span>19.11.22 16:14</span> CET</div>'
        "</a></li>"
    )
    rows = registry_rows(html)
    assert rows == [
        {
            "document_id": None,
            "title": "Provisional Qualifying Classification",
            "publication_cet": "19.11.22 16:14",
            "url": "https://www.fia.com/sites/default/files/decision-document/"
            "2022 Abu Dhabi Grand Prix - Provisional Qualifying Classification.pdf",
            "recalled": False,
        }
    ]


def test_legacy_pdf_cover_binds_number_and_rejects_wrong_event_date():
    row = {
        "document_id": None,
        "title": "Provisional Qualifying Classification",
        "publication_cet": "19.11.22 16:14",
    }
    spec = {"document_id": None}
    item = {"season": 2022, "event_name": "Abu Dhabi Grand Prix"}
    inspected = {
        "document_id": "22",
        "cover_text": "2022 ABU DHABI GRAND PRIX Date 19 November 2022 "
        "Document 22 Title Provisional Qualifying Classification",
    }
    _bind_pdf_identity(row, spec, inspected, item)
    assert row["document_id"] == spec["document_id"] == "22"
    with pytest.raises(ValueError, match="date"):
        _bind_pdf_identity(
            {**row, "document_id": None},
            {"document_id": None},
            {
                **inspected,
                "cover_text": inspected["cover_text"].replace("19 November", "17 November"),
            },
            item,
        )


def test_frozen_audit_replay_preserves_original_evidence_request(tmp_path):
    cutoff = "2025-03-08T15:32:00+00:00"
    candidate = {
        "season": 2025,
        "round": 1,
        "prediction_timestamp_utc": cutoff,
        "circuit_id": "albert_park",
    }
    directory = tmp_path / "data/features/historical_evidence/frozen"
    directory.mkdir(parents=True)
    request = {
        "event": {"circuit_id": "albert_park"},
        "document_bindings": [{"audit_reference": "fia-winter-direct-v2:retained"}],
    }
    original = json.dumps(request).encode()
    (directory / "request.json").write_bytes(original)
    (directory / "targets").mkdir()
    target = {
        "outcomes": {"path": "labels.parquet", "sha256": "labels"},
        "document_bindings": [{"audit_reference": "separate final"}],
    }
    (directory / "targets/labels.json").write_text(json.dumps(target))
    index = tmp_path / "data/benchmarks/gold_core_registry.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "races": [
                    {
                        "event_id": "season=2025/round=01",
                        "prediction_timestamp": cutoff,
                        "features": {"sha256": "frozen"},
                        "outcomes": target["outcomes"],
                    }
                ]
            }
        )
    )
    replay = _resume_request(tmp_path, candidate)
    assert replay is not None
    assert json.loads(replay.read_text())["document_bindings"] == request["document_bindings"]
    assert (directory / "request.json").read_bytes() == original
    assert _resume_request(tmp_path, candidate) == replay
