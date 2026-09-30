"""Later FIA documents must be tied to exact bytes and their actual scope."""

import hashlib

import pytest

from f1_ml_predictor.trust import historical
from f1_ml_predictor.trust.winter import latest_final_record


def _review(tmp_path, monkeypatch, scopes):
    texts = {}
    documents = []
    for index, (scope, title, body) in enumerate(scopes, 1):
        content = f"document-{index}".encode()
        path = tmp_path / f"{index}.pdf"
        path.write_bytes(content)
        texts[path] = {
            "document_id": str(index),
            "cover_text": f"Document {index} Date 19 October 2024 Time 11:00",
            "text": body,
        }
        documents.append(
            {
                "document_id": str(index),
                "cover_document_id": str(index),
                "title": title,
                "url": f"https://www.fia.com/decision/{index}.pdf",
                "scope": scope,
                "artifact": {"path": path.name, "sha256": hashlib.sha256(content).hexdigest()},
            }
        )
    monkeypatch.setattr(historical, "inspect_pdf", lambda path: texts[path])
    return {
        "conclusion": "no_race_classification_change",
        "reviewed_at": "2026-09-30",
        "audit_reference": "exact-fia-later-doc-review",
        "final_publication_cet": "21.10.24 01:36",
        "later_documents": documents,
    }, texts


def test_later_registry_filename_change_still_requires_review():
    final = {
        "document_id": "73",
        "title": "Final Race Classification",
        "publication_cet": "21.10.24 01:36",
        "recalled": False,
        "url": "https://www.fia.com/decision/2024-race-final.pdf",
    }
    later = {
        "document_id": "78",
        "title": "Decision - Right of Review",
        "publication_cet": "24.10.24 17:15",
        "recalled": False,
        "url": "https://www.fia.com/sites/default/files/decision-document/doc_78_-_review.pdf",
    }
    with pytest.raises(ValueError, match="later event documents"):
        latest_final_record([final, later], final["url"], "73")


def test_post_final_review_accepts_rejected_petition_and_organizer_ruling(tmp_path, monkeypatch):
    review, _ = _review(
        tmp_path,
        monkeypatch,
        [
            ("race_review_summons", "Summons - Right of Review", "Petition for a Right of Review"),
            (
                "race_review_rejected",
                "Decision - Right of Review",
                "Right of Review. The petition is rejected.",
            ),
            ("organizer_only", "Decision - Promoter", "Promoter track invasion. A fine remains."),
            ("pre_final_issue", "Final Sprint Starting Grid", "Final Sprint Starting Grid"),
        ],
    )
    historical.verify_post_final_review(review, tmp_path)


def test_post_final_review_rejects_unresolved_or_successful_petition(tmp_path, monkeypatch):
    review, texts = _review(
        tmp_path,
        monkeypatch,
        [
            ("race_review_summons", "Summons - Right of Review", "Petition for a Right of Review"),
            (
                "race_review_rejected",
                "Decision - Right of Review",
                "Right of Review. Petition allowed.",
            ),
        ],
    )
    with pytest.raises(ValueError, match="does not reject"):
        historical.verify_post_final_review(review, tmp_path)
    texts[tmp_path / "2.pdf"]["text"] = "Right of Review. The petition is rejected."
    historical.verify_post_final_review(review, tmp_path)
    review["later_documents"].pop()
    with pytest.raises(ValueError, match="lacks a rejecting decision"):
        historical.verify_post_final_review(review, tmp_path)


def test_post_final_review_rejects_late_issue_and_driver_ruling(tmp_path, monkeypatch):
    review, texts = _review(
        tmp_path,
        monkeypatch,
        [
            ("pre_final_issue", "Post-Qualifying Procedure", "Media procedure"),
            ("organizer_only", "Decision - Promoter", "Car 4 five second time penalty"),
        ],
    )
    texts[tmp_path / "1.pdf"]["cover_text"] = "Document 1 Date 22 October 2024 Time 11:00"
    with pytest.raises(ValueError, match="pre-final issue"):
        historical.verify_post_final_review(review, tmp_path)
    texts[tmp_path / "1.pdf"]["cover_text"] = "Document 1 Date 19 October 2024 Time 11:00"
    with pytest.raises(ValueError, match="event organizer"):
        historical.verify_post_final_review(review, tmp_path)
