"""Offline text fixtures exercising parsing, never publication certification."""

import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.fia_tables import (
    FINAL_INTERMEDIATE_SCHEMA,
    QUALIFYING_SCHEMA,
    ROSTER_SCHEMA,
    ParsingFailure,
    parse_final_text,
    parse_qualifying_text,
    parse_roster_text,
)

EVENT = EventId(2024, 24)
DRIVERS = {
    "Max VERSTAPPEN": "max_verstappen",
    "Max Verstappen": "max_verstappen",
    "Carlos SAINZ": "sainz",
    "Carlos Sainz": "sainz",
    "Liam LAWSON": "lawson",
    "Liam Lawson": "lawson",
    "Sergio PEREZ": "perez",
    "Sergio Perez": "perez",
    "Jack DOOHAN": "doohan",
    "Jack Doohan": "doohan",
}
CONSTRUCTORS = {
    "Oracle Red Bull Racing": "red_bull",
    "Red Bull Racing Honda RBPT": "red_bull",
    "Scuderia Ferrari": "ferrari",
    "Ferrari": "ferrari",
    "Visa Cash App RB F1 Team": "rb",
    "RB Honda RBPT": "rb",
    "BWT Alpine F1 Team": "alpine",
    "Alpine Renault": "alpine",
}
QHEADER = "NO DRIVER NAT ENTRANT Q1 LAPS % TIME Q2 LAPS TIME Q3 LAPS TIME"
RHEADER = "NO DRIVER NAT ENTRANT LAPS TIME GAP INT KM/H FASTEST ON PTS"


def qualifying_row(position, number, driver, team, q1="", q2="", q3=""):
    return f"{position:3} {number:3} {driver:<22} {team:<29} {q1:<30} {q2:<23} {q3}"


def test_qualifying_ignores_percentage_and_wall_clock_and_stops_at_footer():
    first = qualifying_row(
        1,
        1,
        "Max VERSTAPPEN",
        "Oracle Red Bull Racing",
        "1:22.877 6 100.329 18:17:31",
        "1:22.752 6 18:33:17",
        "1:22.207 6 19:00:53",
    )
    second = qualifying_row(
        12,
        55,
        "Carlos SAINZ",
        "Scuderia Ferrari",
        "1:23.178 6 100.693 18:16:17",
        "1:22.804 5 18:32:24",
    )
    text = "\n".join([QHEADER, first, second, "POLE POSITION LAP", first])
    table = parse_qualifying_text(text, EVENT, DRIVERS, CONSTRUCTORS)
    assert table.schema == QUALIFYING_SCHEMA
    rows = table.to_pylist()
    assert len(rows) == 2
    assert rows[0] == {
        "event_id": EVENT.partition(),
        "driver_id": "max_verstappen",
        "constructor_id": "red_bull",
        "position": 1,
        "q1_seconds": 82.877,
        "q2_seconds": 82.752,
        "q3_seconds": 82.207,
    }
    assert rows[1]["q2_seconds"] == 82.804
    assert rows[1]["q3_seconds"] is None


def test_sparse_session_keeps_q3_in_q3_when_q2_is_missing():
    complete = qualifying_row(
        1,
        1,
        "Max VERSTAPPEN",
        "Oracle Red Bull Racing",
        "1:22.877 6 18:17:31",
        "1:22.752 6 18:33:17",
        "1:22.207 6 19:00:53",
    )
    sparse = qualifying_row(
        2,
        55,
        "Carlos SAINZ",
        "Scuderia Ferrari",
        "1:23.178 6 18:16:17",
        "",
        "1:22.408 6 19:00:44",
    )
    rows = parse_qualifying_text(
        "\n".join([QHEADER, complete, sparse]), EVENT, DRIVERS, CONSTRUCTORS
    ).to_pylist()
    assert rows[1]["q2_seconds"] is None
    assert rows[1]["q3_seconds"] == 82.408


def test_qualifying_unclassified_driver_has_no_invented_time_or_position():
    text = "\n".join(
        [
            QHEADER,
            qualifying_row(1, 1, "Max VERSTAPPEN", "Oracle Red Bull Racing", "1:22.877 6 18:17:31"),
            "NOT CLASSIFIED - 107% TIME 1:20.437",
            " 11  Sergio PEREZ      Oracle Red Bull Racing     DNF 2",
            "FASTEST LAP",
        ]
    )
    rows = parse_qualifying_text(text, EVENT, DRIVERS, CONSTRUCTORS).to_pylist()
    assert rows[1]["position"] is None
    assert [rows[1][name] for name in ("q1_seconds", "q2_seconds", "q3_seconds")] == [None] * 3


def test_compact_consecutive_qualifying_rows_and_bwt_entrant_are_supported():
    text = "\n".join(
        [
            QHEADER,
            "1 61 Jack DOOHAN BWT Alpine F1 Team 1:23.456 5 18:10:10 1:23.123 5 18:30:10",
        ]
    )
    row = parse_qualifying_text(text, EVENT, DRIVERS, CONSTRUCTORS).to_pylist()[0]
    assert row["constructor_id"] == "alpine"
    assert row["q1_seconds"] == 83.456
    assert row["q2_seconds"] == 83.123
    assert row["q3_seconds"] is None


def test_entry_list_preserves_race_roster_ignores_superscript_number_and_excludes_p1_only():
    text = "\n".join(
        [
            "Document 11",
            "No. Driver Nat Team Constructor",
            "14   Max Verstappen   NLD   Oracle Red Bull Racing   Red Bull Racing Honda RBPT",
            "553   Carlos Sainz   ESP   Scuderia Ferrari   Ferrari",
            "61   Jack Doohan   AUS   BWT Alpine F1 Team   Alpine Renault",
            "In addition to the list of cars and drivers eligible to take part in the event",
            "part in P1 in accordance with Articles 32.4 and 32.5",
            "37   Unmapped Practice Driver   FRA   Oracle Red Bull Racing   "
            "Red Bull Racing Honda RBPT",
        ]
    )
    table = parse_roster_text(text, EVENT, DRIVERS, CONSTRUCTORS)
    assert table.schema == ROSTER_SCHEMA
    assert table["driver_id"].to_pylist() == ["max_verstappen", "sainz", "doohan"]
    assert table["grid_position"].to_pylist() == [None] * 3
    assert table["start_type"].to_pylist() == ["unknown"] * 3
    assert table["grid_status"].to_pylist() == ["unknown"] * 3


def test_final_classified_dnf_and_unclassified_dnf_do_not_invent_retirement_reason():
    text = "\n".join(
        [
            RHEADER,
            "1 1 Max VERSTAPPEN Oracle Red Bull Racing 58 1:26:33.291 212.246 1:27.438 52 25",
            "17 30 Liam LAWSON Visa Cash App RB F1 Team 55 1:24:36.949 DNF 205.876 1:28.751 52",
            "NOT CLASSIFIED",
            "11 Sergio PEREZ Oracle Red Bull Racing 0 DNF",
            "FASTEST LAP",
            "1 Max VERSTAPPEN Oracle Red Bull Racing 1:22.207 on lap 57 222.002",
        ]
    )
    table = parse_final_text(text, EVENT, DRIVERS, CONSTRUCTORS)
    assert table.schema == FINAL_INTERMEDIATE_SCHEMA
    rows = table.to_pylist()
    assert [row["position"] for row in rows] == [1, 17, None]
    assert [row["classified"] for row in rows] == [True, True, False]
    assert [row["dnf"] for row in rows] == [None, None, None]
    assert [row["dnf_category"] for row in rows] == ["unknown"] * 3
    assert rows[1]["raw_status"] == "DNF"
    assert not any(name in table.column_names for name in ("final_audited", "label_available_at"))


@pytest.mark.parametrize(
    "status, classified, category, dnf",
    [
        ("Finished", True, "finished", False),
        ("Retired mechanical", True, "retired_mechanical", True),
        ("Retired incident", False, "retired_incident", True),
        ("Retired other", False, "retired_other", True),
        ("DNF", False, "unknown", None),
        ("Retired engine", False, "unknown", None),
        ("DNS", False, "did_not_start", None),
        ("DSQ", False, "disqualified", None),
        ("DQ", False, "disqualified", None),
    ],
)
def test_only_explicit_status_taxonomy_produces_known_dnf_targets(
    status, classified, category, dnf
):
    heading = "" if classified else "NOT CLASSIFIED\n"
    prefix = "1 1" if classified else "1"
    text = f"{RHEADER}\n{heading}{prefix} Max VERSTAPPEN Oracle Red Bull Racing 58 {status}"
    row = parse_final_text(text, EVENT, DRIVERS, CONSTRUCTORS).to_pylist()[0]
    assert row["dnf_category"] == category
    assert row["dnf"] is dnf


def test_penalty_marker_after_driver_is_not_part_of_alias():
    text = f"{RHEADER}\n1 1 Max VERSTAPPEN * Oracle Red Bull Racing 58 1:26:33.291"
    row = parse_final_text(text, EVENT, DRIVERS, CONSTRUCTORS).to_pylist()[0]
    assert row["driver_id"] == "max_verstappen"


@pytest.mark.parametrize(
    "parser, text",
    [
        (
            parse_qualifying_text,
            "Driver Q1 LAPS TIME Q2 LAPS TIME Q3 LAPS TIME\n"
            "1 1 Unknown Driver Oracle Red Bull Racing 1:22.877",
        ),
        (parse_final_text, f"{RHEADER}\n1 1 max verstappen Oracle Red Bull Racing 58 1:26:33.291"),
        (
            parse_roster_text,
            "No. Driver Nat Team Constructor\n1 Max Verstappen NLD Unknown Team Unknown Make",
        ),
    ],
)
def test_unmapped_names_fail_without_fuzzy_or_case_insensitive_joins(parser, text):
    with pytest.raises(ParsingFailure, match="unmapped exact"):
        parser(text, EVENT, DRIVERS, CONSTRUCTORS)


@pytest.mark.parametrize(
    "parser, header",
    [
        (parse_qualifying_text, QHEADER),
        (parse_final_text, RHEADER),
    ],
)
def test_duplicate_canonical_driver_and_position_are_rejected(parser, header):
    suffix = "1:22.877" if parser is parse_qualifying_text else "58 1:26:33.291"
    duplicate_driver = (
        f"{header}\n1 1 Max VERSTAPPEN Oracle Red Bull Racing {suffix}\n"
        f"2 1 Max Verstappen Oracle Red Bull Racing {suffix}"
    )
    with pytest.raises(ParsingFailure, match="duplicate canonical driver"):
        parser(duplicate_driver, EVENT, DRIVERS, CONSTRUCTORS)
    duplicate_position = (
        f"{header}\n1 1 Max VERSTAPPEN Oracle Red Bull Racing {suffix}\n"
        f"1 55 Carlos SAINZ Scuderia Ferrari {suffix}"
    )
    with pytest.raises(ParsingFailure, match="duplicate classification position"):
        parser(duplicate_position, EVENT, DRIVERS, CONSTRUCTORS)


def test_dns_or_dsq_in_classified_section_requires_review():
    with pytest.raises(ParsingFailure, match="contradicts"):
        parse_final_text(
            f"{RHEADER}\n1 1 Max VERSTAPPEN Oracle Red Bull Racing 0 DNS",
            EVENT,
            DRIVERS,
            CONSTRUCTORS,
        )


def test_entry_list_tla_is_auxiliary_and_constructor_column_must_be_mapped():
    header = "No. TLA Driver Nat Team Constructor"
    text = f"{header}\n1 VER Max Verstappen NLD Oracle Red Bull Racing Red Bull Racing Honda RBPT"
    row = parse_roster_text(text, EVENT, DRIVERS, CONSTRUCTORS).to_pylist()[0]
    assert row["driver_id"] == "max_verstappen"
    with pytest.raises(ParsingFailure, match="complete name column"):
        parse_roster_text(
            text.replace("Red Bull Racing Honda RBPT", "Unknown Make"),
            EVENT,
            DRIVERS,
            CONSTRUCTORS,
        )


def test_disqualified_section_without_repeated_row_status_is_preserved():
    text = f"{RHEADER}\nDISQUALIFIED\n1 Max VERSTAPPEN Oracle Red Bull Racing"
    row = parse_final_text(text, EVENT, DRIVERS, CONSTRUCTORS).to_pylist()[0]
    assert row["dnf_category"] == "disqualified"
    assert row["dnf"] is None


def test_partial_constructor_alias_cannot_capture_a_longer_name():
    with pytest.raises(ParsingFailure, match="complete timing-sheet entrant"):
        parse_final_text(
            f"{RHEADER}\n1 1 Max VERSTAPPEN Oracle Red Bull Racing 58",
            EVENT,
            DRIVERS,
            {"Oracle": "red_bull"},
        )
    with pytest.raises(ParsingFailure, match="complete name column"):
        parse_roster_text(
            "No. Driver Nat Team Constructor\n1  Max Verstappen  NLD  Oracle Red Bull Racing  "
            "Red Bull Racing Honda RBPT",
            EVENT,
            DRIVERS,
            {"Oracle": "red_bull"},
        )


def test_no_supported_header_empty_tables_and_oversized_text_fail_explicitly():
    for parser in (parse_qualifying_text, parse_roster_text, parse_final_text):
        with pytest.raises(ParsingFailure):
            parser("", EVENT, DRIVERS, CONSTRUCTORS)
        with pytest.raises(ParsingFailure, match="header"):
            parser(
                "1 1 Max VERSTAPPEN Oracle Red Bull Racing 1:22.877", EVENT, DRIVERS, CONSTRUCTORS
            )
        with pytest.raises(ParsingFailure, match="bounded"):
            parser("x" * 2_000_001, EVENT, DRIVERS, CONSTRUCTORS)
    with pytest.raises(ParsingFailure, match="no driver rows"):
        parse_qualifying_text(QHEADER, EVENT, DRIVERS, CONSTRUCTORS)


def test_column_drift_and_excess_qualifying_times_require_review():
    complete = qualifying_row(
        1, 1, "Max VERSTAPPEN", "Oracle Red Bull Racing", "1:22.877", "1:22.752", "1:22.207"
    )
    bad = "2 55 Carlos SAINZ Scuderia Ferrari " + " " * 100 + "1:23.178"
    with pytest.raises(ParsingFailure, match="alignment"):
        parse_qualifying_text(f"{QHEADER}\n{complete}\n{bad}", EVENT, DRIVERS, CONSTRUCTORS)
    with pytest.raises(ParsingFailure, match="more than three"):
        parse_qualifying_text(f"{QHEADER}\n{complete} 1:24.000", EVENT, DRIVERS, CONSTRUCTORS)
