# Scoring evidence audit, 1 October 2026

## Purpose and outcome

Historical Gold has audited finishing positions, but no championship or
constructor points features, because finishing positions alone omit sprint
points, fastest-lap bonuses, reduced-points rules and post-race revisions. This
audit builds the missing evidence layer from official sources:

- `data/audit/scoring_rules.json`: the scoring regime of every season from 2022
  to 2026, bound to retained FIA regulation issues.
- `data/audit/event_points_evidence.json`: points actually awarded, per driver
  and per event, for every completed event from 2022 round 1 to 2026 round 15
  (Azerbaijan, 26 September 2026), with the FIA document chain behind each value.
- `scripts/audit-scoring-evidence.py`: the collector and the deterministic
  builder that produce both files.

Result: **107 of 107 completed events are resolved from FIA documents.** No event
is unresolved and no awarded value was inferred from a classification. All 1,927
current Gold driver-event rows have a matching record.

## Sources and authority order

1. **FIA "Championship Points" documents** (scoring authority). The FIA
   publishes, on each event's document registry, a cumulative matrix of points
   per driver and per entrant for every event of the season. At sprint events,
   a "Championship Points after Sprint" document is also published (from 2023),
   whose event column holds sprint points only. Revised and ICA-revised versions
   appear on the registry of the event they revise. The FIA's printed `F` marker
   next to a race position appears only where a fastest-lap point was awarded.
2. **FIA Sporting Regulations** (2022 to 2025) and **2026 F1 Regulations Section A
   (General Provisions)**: every issue was retained and the scoring articles were
   compared across issues.
3. **FIA stewards' review documents and ICA judgements**, for revisions.
4. **formula1.com** race, sprint and standings pages (official F1, secondary).
   They confirm values, supply sprint splits where no FIA after-sprint matrix
   exists, and supply sprint positions (FIA after-sprint matrices print sprint
   points without positions). They never override an FIA value.
5. **Jolpica** (tertiary, not official): calendar rounds, participation and
   comparison only.

The script retains every response through the repository's immutable,
content-addressed store (`data/raw/fia_audit/objects`,
`data/raw/scoring_audit`). As with existing FIA audit bytes, raw evidence is a
git-ignored local artifact; the outputs record the SHA-256 and URL of every
document used. All retrieved pages are classified `current_state_only` at
capture. The FIA registry publication clock is what certifies timing (see below).

### Matrix parsing and validation

The PDFs are parsed from text coordinates with `pypdf` (the existing audit
dependency): column positions are fitted per document, header codes are aligned
to the calendar with a longest-common-subsequence match (so cancelled events such
as 2023 Imola, and the 2026 calendar change, are skipped correctly), and rotated
pages are normalized. Each parsed document is checked as follows:

- every row's per-event cells must sum to its printed TOTAL. This holds for
  every driver and entrant row of all 123 parsed documents except one (see
  Conflicts);
- no column may carry data for an event raced after the document's publication;
- every data-bearing column must align to a calendar round.

Each record is then cross-checked against the full race scale, the sprint scale,
formula1.com race and sprint points, Jolpica points, and the Gold label position.
Constructor points are checked twice: the sum of the drivers' points against the
FIA entrant matrix (1,085 of 1,085 constructor-event rows agree), and season
totals against formula1.com standings (163 of 163 driver and constructor season
totals agree; all ledger sums equal the latest FIA totals).

## Coverage

Driver-event records by season. "Gold rows" counts records that match a current
Gold row.

| Season | Events | Sprint events | Gold events | Records | Gold rows | Sprint records | FIA sprint split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 22 | 3 | 16 | 440 | 320 | 60 | 0 |
| 2023 | 22 | 6 | 18 | 440 | 360 | 120 | 100 |
| 2024 | 24 | 6 | 23 | 479 | 459 | 120 | 100 |
| 2025 | 24 | 6 | 24 | 480 | 480 | 120 | 114 |
| 2026 (to round 15) | 15 | 5 | 14 | 330 | 308 | 110 | 108 |
| **Total** | **107** | **26** | **95** | **2,169** | **1,927** | **530** | **422** |

- **Audited race point records: 2,169** (one per driver who took part, with
  `race_points`, `bonus_points` and `total_event_points`).
- **Audited sprint records: 530.** Sprint splits come from FIA after-sprint
  matrices (422), formula1.com sprint results (107: the three 2022 sprints, which
  had no FIA after-sprint document; 2023 Azerbaijan, which had none; 2024 United
  States, whose after-sprint document is image-only; and 8 drivers omitted from
  an FIA after-sprint document), or a zero weekend total (1).

Evidence quality by season (`evidence_quality`):

| Season | FIA, formula1.com confirmed | FIA total, formula1.com sprint split | FIA later document, formula1.com confirmed | FIA, unconfirmed | Conflict, secondary source |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 359 | 60 | 20 | 1 | 0 |
| 2023 | 319 | 19 | 99 | 2 | 1 |
| 2024 | 419 | 20 | 40 | 0 | 0 |
| 2025 | 453 | 6 | 20 | 1 | 0 |
| 2026 | 328 | 2 | 0 | 0 | 0 |

"FIA later document" means that the event's own points document does not exist or
is an image-only scan, so the value comes from the next machine-readable FIA
matrix carrying that event's column. The four "unconfirmed" records are drivers
with zero points who appear on no formula1.com race page: Mick Schumacher at 2022
Saudi Arabia (did not start), Sargeant at 2023 Azerbaijan (sprint did not
start), Stroll at 2023 Singapore (withdrawn) and Stroll at 2025 Spain (did not
start). Their FIA cells are blank, read as zero (see below).

## Unresolved events and incomplete documents

No event is unresolved. Ten events lack a machine-readable points document of
their own:

| Event | Reason | Value source |
| --- | --- | --- |
| 2022 Italian GP (round 16) | No Championship Points document on the registry | 2022 Singapore matrix |
| 2024 Mexico City GP (round 20) | No Championship Points document; the registry also lists only a Provisional Race Classification | 2024 Sao Paulo matrices |
| 2023 rounds 2, 6, 7, 13, 19; 2024 round 5; 2025 round 3 | Points PDF is an image-only scan with no extractable text | Next FIA matrix |
| 2024 United States sprint (round 19) | After-sprint PDF is image-only | formula1.com sprint split |

For the seven image-only events the original publication's values were not
transcribed by eye. The later matrix supplies the value; `first_published_at`
therefore reflects the later document, which is conservative for point-in-time
use. Of these ten events, six are in Gold: 2022 round 16; 2023 rounds 2, 6 and 7;
2024 round 5; and 2025 round 3. One recalled document has no retrievable bytes:
2026 Barcelona Doc 69 (14 June 2026, 19:50 CET), superseded by Doc 70 at 20:10.

## Conflicting evidence

All conflicts are listed in `conflicts`. The FIA value is retained in each case.

1. **FIA internal total, 2023 Austrian GP Doc 77 (2 July 2023, 22:05):** Gasly's
   printed TOTAL is 15 while his event cells sum to 16. The per-event cells are
   used; later FIA documents and formula1.com agree with the cells.
2. **formula1.com, 2023 Saudi Arabian GP:** the race page shows 0 points for
   Verstappen (P2 with fastest lap). The FIA matrices and Jolpica give 19 (18 plus
   the fastest-lap point).
3. **FIA document omission, 2025 Chinese GP after-sprint Doc 51:** the PDF
   contains only one driver page (14 of 20 drivers). The six missing drivers'
   sprint points come from formula1.com. One driver is likewise absent from each
   of the 2026 Chinese and 2026 Dutch after-sprint documents.
4. **Blank FIA cells:** the FIA matrix leaves some non-classified retirements and
   non-starters blank (for example Leclerc at 2023 Bahrain, while his 2023
   Australia retirement is printed "NC"). Because every row reconciles to its
   printed total, a blank cell for a listed driver who took part (per formula1.com,
   Jolpica or Gold) is recorded as 0 with adjustment `fia_blank_cell_read_as_zero`.
   13 records.
5. **Regulation text versus effect, 2025 fastest-lap point:** text extraction of
   2025 Issues 2 to 4 still contains the fastest-lap clause, but the rendered PDF
   shows it struck through as a deletion from Issue 2 (17 October 2024). No 2025
   FIA points document awards a fastest-lap point. Any automated reading of
   regulation text must account for revision markup.

## Season-specific scoring and exceptions

Full scales are identical in every season: race 25-18-15-12-10-8-6-4-2-1; sprint
8-7-6-5-4-3-2-1. Constructors score the sum of both cars. Ties in points are
split by count of race wins, then second places, and so on.

| Season | Fastest-lap point | Reduced points trigger | Notes |
| --- | --- | --- | --- |
| 2022 | Yes, top 10 and leader at least 50% | Only a suspended race that cannot be resumed | Sprint sets the race grid. Issue 5 (15 March 2022) fixed reduced column 3 before round 1. |
| 2023 | Yes | Any race shorter than scheduled (Issue 4, 22 February 2023) | Sprint Shootout from Issue 5 (25 April 2023). |
| 2024 | Yes | As 2023 | Sprint Qualifying from Issue 5 (28 February 2024). |
| 2025 | No (deleted in Issue 2, 17 October 2024) | As 2023 | |
| 2026 | No | Section A A2.2.1: laps completed by the leader; two clean consecutive laps required in all cases | Scoring moved to Section A. New final tie-break on qualifying results. |

Observed exceptions in the awarded points:

- **No reduced-points race occurred from 2022 to 2026.** All 2,169 race-point
  values match the full scale for the FIA position. The 2022 Japanese GP ran 28
  of 53 laps but was resumed after suspension, so under the 2022 wording it
  received full points.
- **Fastest-lap bonus:** awarded at 20, 20 and 19 events in 2022, 2023 and 2024.
  At the other 9 events the fastest lap was set outside the top 10 (positions 12
  to 18, checked against Jolpica), so no point was due.
- **Post-race revisions detected from the FIA matrices** (10 records):
  - 2022 United States GP: the 24 October 2022 document gave Alonso 0. The 31
    October 2022 document shows 6, with Vettel 6 to 4, Magnussen 4 to 2, Tsunoda
    2 to 1 and Ocon 1 to 0. No document is titled "Revised"; the change appears
    only in the next matrix.
  - 2026 Monaco GP: Alpine's right of review (Docs 98 and 99) removed Gasly's two
    five-second penalties. Revised Docs 100 and 101 (12 June 2026) gave Gasly 15
    (was 6) and lowered Hadjar, Piastri, Lawson and Lindblad by 3, 2, 2 and 2. ICA
    judgement ICA-2026-06-07-08-09 (3 September 2026) granted the McLaren and Red
    Bull appeals and reinstated the penalties. Docs 106 and 107 (5 September 2026,
    19:15 CET) restore the original values.
  - 2026 Dutch GP: the "Revised following ICA Judgement" Docs 74 and 75 change no
    Dutch values. They re-issue the then-latest cumulative matrices with the Monaco
    column corrected.
- **Pending appeal:** the latest FIA document (2026 Azerbaijan Doc 72) prints
  "Subject to notice of intention to appeal by Audi Revolut F1 Team against
  Stewards' Document 69 (Italian Grand Prix)." Doc 69 took no further action
  against car 22 (Tsunoda). All 22 records of 2026 round 13 carry
  `revision_status = pending_appeal`.
- **Disqualifications in the FIA race cells:** Hamilton and Leclerc (2023 United
  States); Russell (2024 Belgium); Hulkenberg (2024 Sao Paulo); Gasly, Hamilton
  and Leclerc (2025 China); Hulkenberg (2025 Bahrain); Norris and Piastri (2025
  Las Vegas). All were already reflected in the event's own points document.
- **Post-race checks on car 16, 2026 Azerbaijan** (published 30 September 2026):
  the car was found compliant, so there is no change.
- **Right-of-review petitions** at 2023 Saudi Arabia, 2023 Austria, 2024 China,
  2024 United States, 2025 Australia, 2025 Netherlands and 2026 Monaco (Mercedes,
  withdrawn) produced no change between points documents. Any effect they had
  (for example Alonso's restored third place at 2023 Saudi Arabia, decided at
  23:03 CET) predates the event's first points document.

## Schemas

Both JSON files carry `field_definitions`, are sorted, and serialize with sorted
keys. Rebuilding from the same collection index gives byte-identical files.

`scoring_rules.json`: `rules` has one row per season, `event_type`, position
and effective period, with the requested fields `season`, `event_type`,
`position`, `points`, `fastest_lap_rule`, `reduced_points_rule`,
`effective_from`, `effective_to`, `evidence_source` (regulation title, URL and
SHA-256 for the first issue in force and the last issue checked, plus articles)
and `publication_or_rule_date`. `event_type` is `race`,
`race_reduced_col1_2laps_to_lt25pct`, `race_reduced_col2_25_to_lt50pct`,
`race_reduced_col3_50_to_lt75pct`, `sprint` or `fastest_lap_bonus`. Unlisted
positions score zero. `season_rules` holds the per-season narrative: sprint
reduced rule, tie-break, dead heat, constructor scoring, sprint format and every
regulation change found.

`event_points_evidence.json`:

- `records`: one per driver and event, with the requested fields `season`,
  `event`, `event_id_if_known`, `driver`, `constructor`, `race_points`,
  `sprint_points`, `bonus_points`, `penalties_or_adjustments`,
  `total_event_points`, `classification_source`, `scoring_source`,
  `published_at`, `revision_status` and `evidence_quality`. They also carry
  `round`, `in_gold`, `fia_driver_name`, `constructor_source`,
  `race_position_fia`, `race_fastest_lap_marker`, `sprint_position`,
  `sprint_position_source`, `first_published_at`, `points_timeline`,
  `fia_documents_carrying_value`, `sprint_history` and `checks`.
- `constructor_event_points`: FIA entrant points per event, with the sum of
  drivers and the sprint sum.
- `event_summaries`: own, recalled, classification and review documents per
  event; first and latest FIA matrices; fastest-lap count; and appeal notes.
- `documents`: every FIA points document (keyed by `document_key`, the
  SHA-256 of its bytes), including registry and cover identifiers, publication
  times, header alignment, and parse results or failure reasons.
- `season_total_checks`, `conflicts`, `unresolved_events`.

Document references inside `records`, `constructor_event_points` and
`conflicts` are `document_key` strings; resolve them through `documents`.

## Publication times

The FIA registry prints "Published on DD.MM.YY HH:MM CET". As in the existing
FIA audits, it is read as UTC+1. This is the later bound when the wall clock was
actually CEST, so `published_at_utc_upper_bound` never precedes true
publication. One cross-check supports the reading: 2025 China after-sprint Doc 51
has a cover time of 14:05 Shanghai time (06:05 UTC), and the registry shows 07:05.
Document covers (local date and time) are kept alongside.

## What Codex can safely consume

- **Standings and points windows:** use `records[].total_event_points` per
  driver and the matching `constructor` per event. Constructor event points equal
  the sum over that constructor's drivers (verified for every event).
  Race, sprint and bonus components are exact for every record with
  `sprint_points` not null (all 2,169).
- **Point-in-time correctness:** do not use `total_event_points` directly for a
  feature cut at time t. Use the last `points_timeline` entry with
  `published_at_utc_upper_bound <= t`; skip events with no such entry. This
  matters in particular for:
  - the 2026 Monaco revision, whose values in force between 12 June and 5
    September 2026 differ from the final values (cutoffs for 2026 rounds 7 to 12
    must see Gasly at 15);
  - the 2022 United States revision, which reached a points document only on 31
    October 2022, after the Mexico City GP.
- **Gating:** require `driver` not null, `sprint_points` not null and
  `evidence_quality` not starting with `conflict`. The single conflict record
  (2023 Saudi Arabia, Verstappen) has a sound FIA value and may be admitted if
  the gate accepts FIA over formula1.com, which this audit recommends.
- **Rules:** use `scoring_rules.json` only for checks and for future events. All
  historical awarded points come from the evidence file, never from re-applying
  scales.
- **Ties:** standings positions need the countback in `season_rules.tie_break`,
  which counts race placings only (not sprints). 2026 adds a final qualifying
  countback that needs qualifying results.

## Assumptions that remain

1. A blank FIA cell for a listed driver who took part is zero points (13 records).
   This is supported by every row reconciling to its printed total.
2. Sprint positions shown alongside FIA sprint points come from formula1.com.
   Sprint points themselves come from FIA where an after-sprint matrix exists.
3. Registry clock read as UTC+1 (an upper bound).
4. The `F` marker is read as an awarded fastest-lap point only when the FIA
   weekend total exceeds the scale value by exactly one. The data shows no
   counterexample.
5. Constructor identity per driver comes from the Gold roster where available,
   otherwise from formula1.com team names through the existing alias table. FIA
   entrant sums confirm all 1,085 constructor-event assignments.
6. The audit is frozen at 1 October 2026. The 2026 Italian GP appeal is
   unresolved. Re-running `collect` will pick up any later revised document and
   update the timelines.

## Reproduce

```
PYTHONPATH=src python scripts/audit-scoring-evidence.py collect
PYTHONPATH=src python scripts/audit-scoring-evidence.py build data/raw/scoring_audit/collection-<sha>.json
```

The outputs in this commit were built from
`collection-e9673cbbf1f0f97921a2d93f9d1a84d8c536b75cb66ed656420cf25685f9838e.json`.
Model selection logic, Gold benchmarks and the feature pipeline were not
modified.
