"""Binary DNF decisions require independent, unambiguous final statuses."""

from f1_ml_predictor.trust.binary_dnf import audit_binary_status
from f1_ml_predictor.trust.outcomes import DnfCategory


def _fia(raw: str, category: str = "unknown", classified: bool = True) -> dict:
    return {"raw_status": raw, "dnf_category": category, "classified": classified}


def _openf1(dnf: bool = False, dns: bool = False, dsq: bool = False) -> dict:
    return {"dnf": dnf, "dns": dns, "dsq": dsq}


def test_binary_status_requires_three_source_agreement() -> None:
    assert (
        audit_binary_status(
            _fia("DNF", classified=False), {"status": "Retired"}, _openf1(dnf=True)
        )[0]
        == DnfCategory.RETIRED_OTHER
    )
    assert (
        audit_binary_status(_fia("classification_only"), {"status": "Lapped"}, _openf1())[0]
        == DnfCategory.FINISHED
    )
    assert (
        audit_binary_status(_fia("DNF", classified=False), {"status": "Retired"}, _openf1())[0]
        is None
    )
    assert (
        audit_binary_status(_fia("classification_only"), {"status": "Retired"}, _openf1(dnf=True))[
            0
        ]
        is None
    )
    assert (
        audit_binary_status(
            _fia("not_classified", classified=False), {"status": "Lapped"}, _openf1()
        )[0]
        is None
    )


def test_dns_dsq_and_missing_sources_remain_unlabelled() -> None:
    dns = _fia("DNS", "did_not_start", False)
    dsq = _fia("DQ", "disqualified", False)
    assert audit_binary_status(dns, {"status": "Did not start"}, _openf1(dns=True)) == (None, "dns")
    assert audit_binary_status(dsq, {"status": "Disqualified"}, _openf1(dsq=True)) == (None, "dsq")
    assert (
        audit_binary_status(dns, {"status": "Retired"}, _openf1(dnf=True))[1] == "dns_disagreement"
    )
    assert audit_binary_status(_fia("DNF", classified=False), None, _openf1(dnf=True))[0] is None
    assert (
        audit_binary_status(
            _fia("DNF", classified=False),
            {"status": "Retired"},
            {"dnf": True, "dns": True, "dsq": False},
        )[0]
        is None
    )
