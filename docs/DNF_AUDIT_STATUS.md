# Historical DNF evidence audit

The 25 retained FIA final classifications and their exact audited outcome
attachments contain 506 driver-race observations from 24 unique drivers.
`scripts/audit-dnf-coverage.py` verifies the registered outcome and PDF hashes
before reporting their status coverage. The current taxonomy distribution is:

| Category | Driver-race observations | Binary DNF label |
| --- | ---: | --- |
| unknown | 492 | missing |
| did_not_start | 8 | missing |
| disqualified | 6 | missing |
| finished | 0 | false |
| retired mechanical, incident or other | 0 | true |

The PDFs often print `DNF` or `NOT CLASSIFIED`, but neither supplies a supported
retirement cause. A classified position, lap count or gap does not prove a driver
finished. DNS and DSQ are separate outcomes under `audited-dnf-v1` and cannot
be converted to retired DNFs. No binary DNF labels can be promoted from these
sources alone.

An additional exact, versioned official status source with a defensible
publication bound is required for each promoted status. Any new attachment must
keep the original audited classification, race identity, cutoff and outcome
bytes immutable. Historical training can use a new status only if it was
published by that training fold's cutoff.
