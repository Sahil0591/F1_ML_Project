# Legacy FIA Gold recovery, 2026-09-30

The initial five-season audit excluded all 68 completed 2022 to 2024 races at
the identity or registry stage. It reported 51 unsupported registry records,
15 event/season mismatches, and two missing or ambiguous selectors. These were
resolver failures, not evidence that the FIA lacked historical publications.

The recovery accepts both legacy and modern FIA registry rows. Legacy rows can
lack a visible document number and use a different PDF path layout. The resolver
binds each PDF to its registry row using the document cover number, season,
event, issue date, and title, and retains the registry publication time. It
also resolves the Brazilian/Mexican 2022 selector names without treating the
URL shape as proof. Official Formula 1 publication metadata and schedule table
content certify the race clock at the selected cutoff. No race URL is special
cased in the production resolver. The Gold admission rules are unchanged.

The [immutable final report](../data/benchmarks/historical_audit/five-year-coverage-5af93cae7d92870f756dd87aff507520f3500151c8c18873fef3c17d5f76373b.json)
accounts for all 107 completed candidates. All 107 reached direct audit.

| Season | Candidates | Gold | Excluded | Gold driver rows |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 22 | 16 | 6 | 320 |
| 2023 | 22 | 17 | 5 | 340 |
| 2024 | 24 | 15 | 9 | 300 |
| 2025 | 24 | 22 | 2 | 440 |
| 2026, partial | 15 | 14 | 1 | 308 |
| **Total** | **107** | **84** | **23** | **1708** |

There are no Silver or Development admissions. Gold gained 48 historical races
over the initial 36-race audit.

## Recovered historical races

Rounds refer to the season's official race order.

| Season | Gold rounds and events |
| --- | --- |
| 2022 | 01 Bahrain; 03 Australia; 04 Emilia Romagna; 06 Spain; 07 Monaco; 08 Azerbaijan; 09 Canada; 10 Britain; 11 Austria; 12 France; 14 Belgium; 15 Netherlands; 16 Italy; 17 Singapore; 18 Japan; 21 Sao Paulo |
| 2023 | 01 Bahrain; 02 Saudi Arabia; 04 Azerbaijan; 05 Miami; 06 Monaco; 07 Spain; 08 Canada; 09 Austria; 10 Britain; 11 Hungary; 12 Belgium; 14 Italy; 16 Japan; 17 Qatar; 20 Sao Paulo; 21 Las Vegas; 22 Abu Dhabi |
| 2024 | 01 Bahrain; 02 Saudi Arabia; 06 Miami; 07 Emilia Romagna; 08 Monaco; 09 Canada; 10 Spain; 11 Austria; 12 Britain; 13 Hungary; 16 Italy; 17 Azerbaijan; 18 Singapore; 23 Qatar; 24 Abu Dhabi |

## Remaining exclusions

| Reason | Count | Events |
| --- | ---: | --- |
| Later event documents require final outcome review | 8 | 2022 Hungary, United States, Abu Dhabi; 2023 Australia; 2024 Japan, China, United States; 2026 Canada |
| Outcomes do not match the supplied field roster | 5 | 2022 Saudi Arabia; 2023 Singapore, United States; 2024 Belgium; 2025 Spain |
| Entry-list constructor alias does not match a complete name column | 3 | 2022 Miami; 2024 Australia, Netherlands |
| Unsupported qualifying or race table header | 3 | 2023 Netherlands, Mexico City; 2025 Austria |
| Cutoff-valid official race schedule missing | 2 | 2024 Sao Paulo, Las Vegas |
| Published local time contradicts printed GMT | 1 | 2022 Mexico City |
| Final race classification missing | 1 | 2024 Mexico City |

The two 2024 timetable exclusions reflect schedule content that conflicts with
the current schedule hint. The 2022 Mexico City season article likewise prints
incompatible local and GMT start times. Neither conflict was resolved by
silently trusting the current API clock. The final classification and post-race
review exclusions remain pending exact official outcome evidence.

## Rebuilt benchmark and frozen evaluation

The [Gold Core rolling benchmark](../data/benchmarks/gold_core_rolling_v1/18739f95d2bc254f593117f36323ad9bb367c517162564a44a6d1121ab12c908)
contains 84 races and 1708 driver-race rows. Rolling finish features use only
contiguous, audited, same-season prior races with labels available by cutoff.
Other missing feature families were not filled from current-state data. The
coverage report records each feature count by season.

The [frozen chronological backtest](../models/backtests/dataset-02f02c1aa11c7227/gold.json)
evaluated 82 events and 1668 driver-race rows after two initial events lacked
enough prior training labels. The logistic winner log loss was 1.2750 versus
1.6696 for the heuristic; position MAE was 2.7383 versus 2.8023. The
[model comparison](../models/experiments/gold/dataset-02f02c1aa11c7227/comparison.json)
evaluated 81 paired events per backend and left model selection deferred due
to insufficient paired events or documented baseline regressions. No backend
was promoted.

Reproduce discovery and audit with `scripts/expand-historical-gold.py`.
The `--discovery` argument reuses the immutable discovery catalog. The
[workflow](GOLD_WORKFLOW.md) documents the corresponding benchmark and
evaluation commands.
