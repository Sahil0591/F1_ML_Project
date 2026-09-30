# Historical FIA Gold recovery, 30 September 2026

The exhaustive direct audit covered all 107 historical candidates. It admitted
95 Gold races, compared with 84 before this recovery. The [immutable coverage
report](../data/benchmarks/historical_audit/five-year-coverage-433145cf46239d047e7f6a0ccc2a8dd79bad87de91fd1a258afdc238bd1e38e0.json)
records 16 of 22 in 2022, 18 of 22 in 2023, 23 of 24 in 2024, 24 of 24 in
2025, and 14 of 15 in 2026. No Gold evidence requirement was changed.

## 2025 recovery

| Race | Result | Exact evidence and interpretation |
| --- | --- | --- |
| Spain | Gold | [FIA event registry](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2025-2071/event/Spanish%20Grand%20Prix) includes the later approved Car 18 withdrawal and [final race classification](https://www.fia.com/system/files/decision-document/2025_spanish_grand_prix_-_final_race_classification.pdf). Stroll remains in the pre-race feature roster. The later approved withdrawal creates a DNS target with unknown binary DNF status and outcome-only provenance. |
| Austria | Gold | The [FIA event registry](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2025-2071/event/Austrian%20Grand%20Prix) records recalled qualifying Doc 29 and selected replacement Doc 31. The [final race Doc 48](https://www.fia.com/system/files/decision-document/2025_austrian_grand_prix_-_final_race_classification.pdf) has an image-only classification table. Its visually audited, SHA-bound [transcription](HISTORICAL_IMAGE_FINAL_REVIEWS.json) contains 16 classified drivers, three DNF and one DNS. |

The complete audit confirms 2025 reached 24 of 24 Gold, with no remaining
2025 exclusion.

## Nine investigated 2024 exclusions

| Race | Result | Investigation |
| --- | --- | --- |
| Australia | Gold | The [FIA event registry](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043/event/Australian%20Grand%20Prix) supplies the pre-cutoff Car 2 withdrawal notice. Qualifying correctly establishes a 19-driver predictive roster. Versioned 2024 constructor aliases reconcile entry-list and timing-sheet names. |
| Japan | Gold | The [final classification](https://www.fia.com/sites/default/files/decision-document/2024%20Japanese%20Grand%20Prix%20-%20Final%20Race%20Classification.pdf) is followed in the registry by a post-qualifying media procedure whose PDF is dated before the race. Its exact PDF and publication metadata are bound in the [later-document review](HISTORICAL_2024_LATER_REVIEWS.json). |
| China | Gold | The [later FIA right-of-review decision](https://www.fia.com/sites/default/files/decision-document/doc_80_-_2024_chinese_grand_prix_-_decision_-_aston_martin_right_of_review_0.pdf) concerns the Sprint penalty and Sprint classification. The stewards dismissed the petition. The two summonses and decision are bound to retained PDFs and the registry in the [review](HISTORICAL_2024_LATER_REVIEWS.json); the race final remains authoritative. |
| Belgium | Gold | The [later final classification](https://www.fia.com/sites/default/files/decision-document/2024%20Belgian%20Grand%20Prix%20-%20Final%20Race%20Classification.pdf) begins with Russell's disqualification. The parser keeps that DSQ distinct from DNF and preserves the final finishing order. |
| Netherlands | Gold | The 2024 constructor aliases reconcile the [FIA event registry](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043/event/Dutch%20Grand%20Prix) entry list and classifications. |
| United States | Gold | The [McLaren right-of-review decision](https://www.fia.com/sites/default/files/decision-document/doc_78_-_2024_united_states_grand_prix_-_decision_-_mclaren_-_right_of_review.pdf) rejected the petition about Norris's five-second penalty. Later [COTA](https://www.fia.com/sites/default/files/decision-document/usa_doc_84_-_decision_-_cota_-_right_of_review.pdf) and [US Race Management](https://www.fia.com/sites/default/files/decision-document/usa_doc_85_-_decision_-_usrm_-_right_of_review.pdf) reviews concern track access and the promoter fine. Thirteen later registry rows are bound in the [review](HISTORICAL_2024_LATER_REVIEWS.json), including pre-race documents listed late by the registry. |
| Mexico City | Excluded | The [FIA decision registry](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043/event/Mexico%20City%20Grand%20Prix) has provisional race classification Doc 46 but no final race classification. The [archived FIA race page](https://www.fia.com/events/fia-formula-one-world-championship/season-2024/mexico-city-grand-prix/race-classification) displays results without an auditable historical publication time or version chain. Exact exclusion: `final_race_classification_missing`. A current archived page alone does not meet the final outcome contract. |
| São Paulo | Gold | The pre-cutoff [FIA Change to Timetable](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043/event/S%C3%A3o%20Paulo%20Grand%20Prix) Doc 48 approved Version 5. It supersedes the conflicting discovery clock and fixes the race start at 15:30 UTC on 3 November. The [official event timetable](https://www.formula1.com/en/latest/article/formula-1-lenovo-grande-premio-de-sao-paulo-2024-timetable.019DEsMvCw1OHYxrQkKacx) independently shows 12:30 local. |
| Las Vegas | Gold | The [official event timetable](https://www.formula1.com/en/latest/article/formula-1-heineken-silver-las-vegas-grand-prix-2024-timetable.1RlJ0Pdt7DuwNUBTBynj3d) was published before cutoff. Its Saturday 23 November 22:00 local race start is Sunday 24 November 06:00 UTC. The stale discovery UTC date was corrected from the published timetable. |

The 2024 direct audit therefore recovered eight of the nine previously excluded
races and reached 23 of 24 Gold. The Mexico City exclusion remains explicit.

## Generic implementation and evidence boundaries

- Recalled FIA documents remain in the exact registry chain and cannot be selected when a valid replacement exists. The final resolver examines the whole event registry, including later documents whose filenames use a different path or naming style.
- Versioned constructor aliases identify unambiguous contemporary variants. They do not assert roster membership. Pre-cutoff withdrawal notices can explain a smaller qualifying field. A later approved withdrawal is attached only to outcome provenance, with DNS separate from DNF.
- Final table parsing supports a leading disqualification, `NOT CLASSIFIED` sections, and explicit DNF, DNS and DSQ. An image-only final table requires a completed visual audit bound to the exact PDF SHA. Parsed final text cannot be overridden by transcription.
- Official event timetable rows resolve local dates across UTC midnight. A published FIA timetable amendment can supersede the original schedule only if its PDF, registry publication time, approval text and race row verify before the prediction cutoff.
- A later-document review is exact-registry and exact-PDF bound. Summonses require a rejecting decision. Organizer rulings, pre-final issue dates and rejected race or Sprint reviews have separate verification rules. New or changed later documents block Gold until reviewed.
- FIA outcome documents, decision publication timestamps and historical feature cutoffs remain separate. Missing optional feature families and unknown DNF causes were not filled using current-state data.

## Rebuilt artifacts and evaluation

The [Gold Core rolling benchmark](../data/benchmarks/gold_core_rolling_v1/4c0cae1bc87191ef5e3b78eed2b6228396e4efec7ae8e7028651712e53de5412)
contains 95 races and 1,927 driver-race rows. The frozen [chronological
backtest](../models/backtests/dataset-6d854400f69fc0c0/gold.json) evaluated
93 events and 1,887 prediction rows after initial training exclusions. Logistic
winner log loss was 1.3667 versus 1.7067 for the heuristic. Logistic position
MAE was 2.7319 versus 2.7879. The [model
comparison](../models/experiments/gold/dataset-6d854400f69fc0c0/comparison.json)
also completed; backend selection remains deferred under the existing gate.

The focused parser, schedule, alias, image review and later-decision tests cover
the new edge cases. The full suite and lint checks were run before commit.
