# Five-season Gold Core expansion baseline, 2026-09-30

This records the initial audit before legacy FIA recovery. The completed
[recovery report](LEGACY_FIA_GOLD_RECOVERY_2026-09-30.md) supersedes its coverage
and exclusion counts.

The exhaustive discovery window covers completed 2022–2026 races as of the
discovery clock. Every completed schedule identity has an individual result in
the [immutable coverage report](../data/benchmarks/historical_audit/five-year-coverage-d8c2720116bae4a98324788f3deeba85098cede6c179d7bfba88bcd3421ac37f.json).
The current 2026 season is partial. No minimum-race target stopped the audit.

| Season | Candidate races | Gold | Silver | Development | Excluded | Gold driver rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 22 | 0 | 0 | 0 | 22 | 0 |
| 2023 | 22 | 0 | 0 | 0 | 22 | 0 |
| 2024 | 24 | 0 | 0 | 0 | 24 | 0 |
| 2025 | 24 | 22 | 0 | 0 | 2 | 440 |
| 2026 | 15 | 14 | 0 | 0 | 1 | 308 |
| **Total** | **107** | **36** | **0** | **0** | **71** | **748** |

## Feature coverage

Counts are nonmissing driver rows in the Gold benchmark. All grid, forecast,
practice, tyre, standings, circuit and original historical aggregate features
remain missing because cutoff-valid direct evidence was not bound. They were
not filled from current API state. The separate rolling builder verified
same-season contiguous prior-race windows using exact audited labels.

| Season | Qualifying position / time | Teammate qualifying delta | Rolling finish mean 3 / 5 / 10 | Rolling DNF rate 3 / 5 / 10 |
| --- | ---: | ---: | ---: | ---: |
| 2022–2024 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 2025 | 432 / 440 | 426 / 440 | 299 / 219 / 60 | 0 / 0 / 0 |
| 2026 | 303 / 308 | 298 / 308 | 170 / 109 / 0 | 0 / 0 / 0 |

The complete per-feature, per-year matrix and prior-window completeness counts
are in the coverage JSON.

## Evidence coverage by source

| Evidence checkpoint | Races |
| --- | ---: |
| Jolpica schedule identity, current state only | 107 |
| Exact FIA event registry metadata | 39 |
| FIA qualifying, grid and final publication metadata | 39 each |
| FIA penalty publication metadata | 1 |
| Race-grid publication metadata by selected cutoff | 24 |
| Decision or infringement metadata by selected cutoff | 39 |
| Direct-audited FIA qualifying table and final outcome | 36 each |
| Direct-audited Formula 1 schedule | 36 |

Registry metadata and current schedule identities never grant Gold. The
2025–2026 direct audit attempted all 39 races, including the six that failed.
The coverage JSON lists exact grid and decision document numbers available by
each cutoff. Grid and separate penalty values were left missing; the latest
required qualifying classification was audited for each admitted Gold race.
The other 68 races were rejected at the official selector/registry identity
stage: the existing direct-audit parser cannot bind the older registry format
to exact publication document numbers and contemporary schedule proofs. Their
PDF tables were not certified. No Silver or Development tier was inferred from
unverified metadata.

## Exclusion reasons

| Exact reason | Races |
| --- | ---: |
| FIA registry contains no supported exact publication records | 51 |
| Registry document identity does not match exact event and season | 15 |
| Exact FIA selector is missing or ambiguous | 2 |
| Later event documents require final outcome review | 1 |
| Outcomes do not match the supplied field roster | 1 |
| Supported qualifying/race table header was not found | 1 |

The last three are 2026 Canada, 2025 Spain and 2025 Austria respectively.
Canada's later document imposes a ten-second penalty and needs an exact
post-decision final classification review. The [later-document reviews](HISTORICAL_2026_LATER_REVIEWS.json)
bind Austria 2026's no-penalty decision and the Dutch and Italian media
procedures to exact PDF hashes. Those three races passed the unchanged Gold
builder after the review. Gold now has 14 of 15 completed 2026 races.

The report retains a reason and evidence flags for every rejected race.
`scripts/expand-historical-gold.py` repeats discovery and direct auditing;
`--discovery PATH` reuses the retained discovery catalog for an exact replay.
