# Championship scenario simulation

Phase 9 provides offline Monte Carlo infrastructure through
`f1_ml_predictor.simulation.simulate_championship`. Local fixtures verify
engineering behavior; they do not establish F1 predictive accuracy.

## Input contract

`CurrentStandings` contains canonical Jolpica driver and constructor IDs,
complete current point totals, an availability timestamp, and a source SHA-256.
Every remaining entrant must appear explicitly, including a zero-point driver.
Existing constructor totals are independent of the current driver assignments,
so replacements and transfers do not reallocate old constructor points. Known
sanctions can be represented in the supplied current totals.

Each `EventSimulation` supplies its canonical season/round, scheduled timestamp,
availability timestamp, complete entered roster with event-specific constructor
assignments, model ID, source hash, and complete joint sampled orders. Every
sample must contain every entered driver exactly once. Repeated orders preserve
their empirical probability mass. The simulator resamples one entire order per
event; it never draws driver positions independently from marginal probabilities.
Availability timestamps must be UTC and no later than the prediction cutoff.
Remaining events must be scheduled after that cutoff.

Every order also requires an explicit `points_eligible_samples` driver subset.
This subset travels with the order during resampling. PL/DNF draws alone do not
predict FIA classified status, laps completed, DNS or DSQ. A retirement does not
automatically establish points eligibility or ineligibility. Engineering callers
may explicitly select `all_entered_engineering_assumption` and supply the complete
roster in each eligibility sample. This policy blocks validated scenario status.
An ineligible driver gets zero points at its supplied order position; later
drivers are not silently moved up. Callers must supply the final classification
order when penalties or exclusions change point-awarding positions.

## Explicit scoring cases

There are no default points tables. `PointsRules` requires the season, `race` or
`sprint`, points by position, an identifying rules ID, and source URL. Supported
seasons are 2019 through 2026. Sprint cases before 2021 are rejected. The caller
must select `full`, `explicit_shortened`, or `no_points` and supply the exact
table for that scenario. Unknown distance regimes are rejected. The simulator
does not infer completed laps, green-flag laps, shortened-event thresholds,
cancellation probabilities or the applicable regulations issue. `no_points`
requires a zero table and no bonus.

For 2019-2024 race scenarios, an optional one-point fastest-lap bonus requires
an explicit top-ten eligibility limit and one fastest-lap driver, or `None`,
per order. Only an eligible driver in the top ten receives it; the same point
goes to that driver's constructor. Sprint bonuses and bonuses from 2025 onward
are rejected. FIA 2025 article 6.4 contains the race and sprint tables without
the former fastest-lap bonus. Articles 6.5-6.6 specify separate shortened-race
and shortened-sprint cases. See the [FIA 2025 sporting regulations, issue 5](https://www.fia.com/system/files/documents/fia_2025_formula_1_sporting_regulations_-_issue_5_-_2025-04-30.pdf)
and the [FIA World Motor Sport Council decision](https://www.fia.com/news/future-regulations-across-multiple-categories-confirmed-during-world-motor-sport-council).

For 2026, scoring is in Section A rather than Section B. The official
[2026 Section A, issue 3, dated 2026-06-25](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_03_-_2026-06-25.pdf),
articles A2.2.1-A2.2.3, specifies race and sprint distance-dependent tables with
no fastest-lap point. The reviewed PDF SHA256 is
`6ce8c9420ddb8194e1b5ea8f678dbbdcb4bb2094780e636324b709ee70d276d6`.
Callers must still supply the exact table for the selected distance case.

Tables are explicit caller inputs, not certification that the named case
matches official regulations. No automatic historical rules selection is made.
Historical dropped-score seasons and future rule changes outside the supported
range require a new contract and review.

## Results and limits

Results include separate WDC and WCC probabilities, mean final points,
final position distributions, and unresolved top-points tie mass. Entities tied
on points share the tied positions equally in the position distribution, so
every row and position column sums to one without invented countback. `title_probability` means a sole highest-points
finisher. `tied_for_title_probability` records each entity's participation in
an unresolved tie. For either title, sole-title probabilities plus the single
`unresolved_tie_probability` sum to one. Participation probabilities can sum
above one because a tie includes multiple entities. Current point totals do not
contain historical finish counts, so the simulator does not invent FIA countback.
Exact rational point arithmetic preserves fractional-point ties.

Remaining events, including the race and sprint at one round, are independently
resampled. Shared season-level form, reliability, weather and incident effects
are omitted. Both cars' points contribute to constructor scores, preserving the
within-event dependencies represented by the supplied joint orders. These
assumptions need validation before interpreting scenario probabilities as useful
forecasts.

Fixed seeds and canonical event ordering make repeat runs deterministic. The
standard-error bound `0.5 / sqrt(simulations)` concerns sampling error only;
it is not model uncertainty or an accuracy guarantee. No GPU is required.

`to_dict()` returns a JSON-compatible result with a canonical result SHA-256.
`write_json(Path(...))` persists it. Provenance binds complete input content,
standings, joint samples, classification samples, scoring cases, declared source
hashes, model validation evidence, seed, sample count, and simulator source hash.
The result hash excludes its own `sha256` field. Declared source hashes identify
upstream evidence; callers remain responsible for auditing those source files.

The default status is `engineering_only`. A `ValidationEvidence` request with
`model_validated=True` must have Gold evidence from at least 25 distinct
independent races available by the cutoff, must bind every remaining event's
model ID, and must use explicit classification eligibility. Insufficient or
incompatible validation raises an error rather than promoting a fixture. The
gate also rejects future seasons and same-season validation rounds at or after
the first remaining prediction event, even when their availability is claimed.
simulator checks this contract, not the accuracy findings in an external report.
Passing the gate produces `validated_model_scenario`; it does not independently
validate championship calibration. Silver and Development stay engineering or
exploratory evidence.
