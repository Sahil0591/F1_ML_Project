# Provisional-grid cohort, 11 October 2026

## Why a separate cohort

The [historical enrichment](GOLD_HISTORICAL_ENRICHMENT_2026-10-01.md) left grid
values on only 24 Gold races. All 57 Gold races from 2022 to 2024 have an FIA
Provisional Starting Grid, but it was published about 1 to 26 hours after the
post-qualifying cutoff. Filling those values into the post-qualifying cohort
would leak penalties decided after its cutoff. The frozen cutoff therefore stays
unchanged; the grids instead define a new, separately timed cohort.

## Audit

`scripts/audit-historical-grids.py --cohort provisional_grid` gives each Gold race
its own cutoff. It is one minute after the first non-recalled Grand Prix starting
grid in the retained FIA registry. It is never earlier than the post-qualifying
cutoff and must precede the audited Gold race start. The grid known at that minute
is parsed from the retained PDF.

The audit is
`provisional-grid-audit-fca5b249cf68c3405d9ed244ea1432d4649cd143dfbd1075adcdb1ffd46b5301.json`.
It covers all 96 Gold races and audits 95. All 57 races from 2022 to 2024 are
included. 2026 Australia remains missing because its provisional grid gives no
slot to three entrants permitted to start. Grid values now cover 1,926 of 1,949
contract rows, compared with 492 before.

Three resolver gaps were fixed. Legacy races use the registry page retained by
discovery, and their rows may lack the document number printed on the PDF cover.
Race grids are matched by exact title, so a stewards' "Starting Grid
Infringement" decision is no longer treated as a grid. The parser also rejoins
grids printed one cell per line (2022 Japan) and reads drivers listed under a
pit-lane heading (2022 Emilia Romagna). Every previously audited PDF reparses
identically. The post-qualifying audit rebuilds to the same hash,
`7dd85b9415d9a1cfa80a58d4ab15118feeac684539854a854995d3dc08ab13dd`.

The retained Jolpica schedule places the 2024 Las Vegas race at
2024-11-23 06:00 UTC. The audited formula1.com timetable gives 2024-11-24 06:00 UTC,
so the audit bounds cutoffs with the audited race start.

## Contract and features

`provisional_grid` is addendum `grid-addendum-v1` to protocol v3. The four frozen
v3 cutoffs, their datasets and the protocol hash are unchanged. The contract keeps
every `pre_race` predictor, recomputes history at the later cutoff, and adds two
features:

- `grid_places_lost`: grid position minus qualifying position. For a pit-lane
  start, it is entrants plus one minus qualifying position.
- `pit_lane_start`: 1 for a pit-lane starter and 0 otherwise. A pit-lane start no
  longer appears to be missing data.

The cohort has 48 drops of five or more places against qualifying, compared with
9 previously available.

## Evaluation

The contract was evaluated with protocol v3 on CPU with seed 42. Both contracts
have 93 outer races. Deltas are the selected pipeline's `provisional_grid` loss
minus its `pre_race` loss, with race-bootstrap 95% intervals.

| Metric | provisional_grid | pre_race | Paired delta |
| --- | ---: | ---: | ---: |
| Winner log loss | 1.2813 | 1.2315 | +0.050 [+0.007, +0.101] |
| Podium Brier | 0.07206 | 0.07221 | -0.0002 [-0.0011, +0.0009] |
| Position MAE | 2.662 | 2.669 | -0.007 [-0.020, +0.008] |

The winner regression is significant. Most of it comes from 2024 Miami and from
2022 Belgium, where Verstappen started 15th and won. Podium and position
intervals include zero.

Forty-two classified drivers lost five or more grid places. The new contract
expected them to finish 1.6 places behind qualifying, compared with 1.0 under
`pre_race`. They actually finished 0.1 places ahead on average. Its
mean absolute error on those drivers was 2.85, compared with 2.61. The 15 penalised
drivers who qualified in the top five finished 2.1 places behind qualifying on
average, despite larger grid drops. The other 27 finished 1.3 places ahead of
qualifying and 8.9 ahead of their grid slots. In this Gold history, qualifying
pace explains finishing order better than the penalised grid slot.

## Decision

The live race prediction keeps the `pre_race` and `post_qualifying` contracts. A
live FIA grid capture would only feed a contract that scored worse, so it was not
added. A fixed hand-set grid-penalty adjustment would also move penalised drivers
further from the observed finishes. No gate or validated forecast changed. The
cohort, audit and comparison remain available for re-evaluation as Gold grows.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/audit-historical-grids.py data/benchmarks/historical_audit/five-year-coverage-bfce9f61de1261786f263ab29547fdc152e6a7649dfec9d8dc7454bc9ffc40c8.json --cohort provisional_grid
.\.venv\Scripts\python.exe -m f1_ml_predictor evaluate-cutoffs --contracts provisional_grid pre_race
.\.venv\Scripts\python.exe scripts/compare-grid-contract.py
```

`load_audited_history` selects the newest provisional-grid audit that covers
exactly the current Gold races. It replays the registry binding and PDF parse
before using it. The comparison is written beside the evaluation as
`comparison-<grid-audit-sha>.json`, including every penalised driver.
