# Binary DNF outcome audit

The original 29 retained FIA final classifications and outcome attachments
contain 594 driver-race observations from 24 unique drivers. They remain
immutable. The original cause-oriented taxonomy yields 580 unknown, eight DNS,
six DSQ and zero known binary DNF labels.

`binary-dnf-v1` is a separate retrospective target audit. It cross-checks each
registered FIA final row with [OpenF1 session_result](https://openf1.org/docs/)
`dnf`, `dns` and `dsq` fields and the retained Jolpica race result `status`.
Exact API response bytes, retrieval metadata, hashes, session identities, driver
numbers, the original outcome hash and every row-level decision are retained
under ignored `data/raw/dnf_audit_v1` and
`data/benchmarks/gold_core_binary_dnf_v1`. The source API observations are
classified `current_state_only`; none is a predictive input.

| Decision | Driver-race observations | Binary target |
| --- | ---: | --- |
| FIA `DNF`, Jolpica `Retired`, OpenF1 `dnf=true` | 67 | true |
| FIA classified row, Jolpica `Finished` or `Lapped`, OpenF1 all flags false | 480 | false |
| DNS | 8 | unknown |
| DSQ | 6 | unknown |
| OpenF1 row or driver mapping missing | 29 | unknown |
| Other source disagreements or ambiguous FIA row | 4 | unknown |

The new outcome version retains the FIA `raw_status`, classified position,
original final-document binding and original label availability bound. It uses
`retired_other` for agreed DNF because a mechanical or incident cause is not
established. DNS and DSQ remain distinct, and disagreements are never coerced.
The row-level audit includes one DSQ cross-check disagreement among the six DSQ
rows. A versioned catalog points to new outcome Parquet files; the original
registry, snapshots and outcomes are preserved. The derived rolling benchmark
uses the same contiguous prior-race and label-clock rules.

The binary target describes an outcome established after the race. It does not
need pre-race feature availability evidence. The historical training clock is
the independently audited FIA final-classification publication bound. OpenF1
and Jolpica corroborate the retrospective binary interpretation, but their
retrieved current versions do not prove their exact historical publication
times. This is a provenance limit for the target, not a claim that those current
responses were available before a prediction cutoff. Any later source correction
requires a new audit and outcome version.

The final 29-race comparison evaluates DNF on 494 paired test driver rows. DNS,
DSQ and unknown rows are excluded from DNF fitting and scoring. Cause
classification remains deferred. See [regression audit](BASELINE_REGRESSION_AUDIT.md).
