# Gold chronological evaluation protocol v1

Frozen before expanding the eight-race Gold Core benchmark. The code embeds a
canonical SHA-256 of its protocol settings in each comparison report. Any change
to a threshold, metric, split, or selection rule requires a new protocol version.

- Evaluate Gold only for primary accuracy. Keep cutoff kinds separate.
- Form complete race groups in event chronology. Training and calibration labels
  must be available by their respective next prediction cutoffs. Keep every
  driver's observations from a race together.
- Compare candidates and heuristic/logistic baselines on the same paired race
  groups and the same driver-race observations. Count unique races for sample
  thresholds; report driver-race observations separately from unique drivers.
- Reserve earlier eligible races for calibration, then evaluate the later outer
  race. Fit imputation and model parameters only on earlier fit races.
- Use at least 15 distinct paired Gold test races for a preliminary comparison.
  Report all task sample counts, missing DNF labels, and uncertainty. No model is
  selected at this stage.
- Require at least 25 distinct paired Gold test races from at least 25 eligible
  races for provisional model selection. Every task metric must be evaluated, and
  a candidate must have no recorded regression against either paired baseline in
  the existing winner, podium, DNF, and position checks. Among passing candidates,
  select the lowest winner log loss. Future independent races must confirm it.
- Preserve prediction snapshots, benchmark bytes, fold membership, source hashes,
  model configuration, seed, calibration settings, and this protocol hash in the
  report. Changes to audited labels create new versions; past predictions remain
  immutable.

The protocol defines minimum evidence and a conservative selection rule. It does
not assert that 15 or 25 races guarantee precise estimates. A championship
scenario stays an engineering result until its race model has passed the Gold
selection and independent confirmation gates.
