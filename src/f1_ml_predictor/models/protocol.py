"""Frozen minimum evidence thresholds for chronological Gold comparisons."""

import hashlib
import json

PROTOCOL_VERSION = "gold-chronological-v2"
PRELIMINARY_PAIRED_EVENTS = 15
SELECTION_PAIRED_EVENTS = 25

PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "tier": "Gold",
    "cohort": "one named cutoff kind at a time",
    "outer_split": "complete events in chronological rolling origins",
    "label_gate": "all training labels available by the next prediction cutoff",
    "paired_comparison": "identical event and driver observations for every backend and baseline",
    "minimum_preliminary_paired_events": PRELIMINARY_PAIRED_EVENTS,
    "minimum_selection_paired_events": SELECTION_PAIRED_EVENTS,
    "selection": (
        "task-specific Gold selection requires 25 paired races, all task metrics evaluated, "
        "no task baseline regressions and race-bootstrap primary-loss improvement "
        "versus both baselines"
    ),
    "confirmation": "future independent races required after provisional selection",
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
