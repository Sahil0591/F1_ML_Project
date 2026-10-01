"""Readable markdown for one cutoff-specific development prediction run."""

from __future__ import annotations

from typing import Any


def probability(value: float | None, draws: int | None = None) -> str:
    """Keep small probabilities visible instead of rounding them to 0.0%."""
    if value is None:
        return "n/a"
    if value == 0:
        return f"0 of {draws} draws" if draws else "0"
    if value < 0.001:
        return f"{100 * value:.2g}%"
    return f"{100 * value:.1f}%"


def coarse(value: float) -> str:
    """Championship precision limited to whole percentage points."""
    if value == 0:
        return "0 in simulation"
    if value < 0.01:
        return "<1%"
    if value > 0.99:
        return ">99%"
    return f"{round(100 * value)}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _metric_row(name: str, metrics: dict[str, Any]) -> list[str]:
    winner, podium, finish = metrics["winner"], metrics["podium"], metrics["finishing_position"]
    sharp = metrics.get("sharpness", {})
    return [
        name,
        _num(winner["log_loss"]),
        _num(winner["brier_score"]),
        _num(winner["top_1_accuracy"], 2),
        _num(podium["brier_score"], 4),
        _num(finish["mean_absolute_error"], 2),
        _num(winner["ece"]),
        _num(podium["ece"]),
        str(winner["winner_below_1_percent"]),
        _num(sharp.get("effective_win_contenders"), 2),
    ]


_INPUT_LABELS = {
    "recent_finish_mean_3": "3-race finish mean",
    "driver_finish_mean_any_10": "10-race finish mean",
    "driver_qualifying_mean_any_5": "5-race qualifying mean",
    "constructor_average_finish_last_5": "team 5-race finish mean",
}


def _movement_narrative(comparison: dict[str, Any], championship: dict[str, Any]) -> list[str]:
    """Explain every win movement of at least three points and the title movement."""
    lines = [
        "What changed overall: the pre-weekend contract adds cross-season driver finish and "
        "qualifying form, driver and constructor reliability and circuit history to the "
        "3/5/10-race form that dominated c52b674; the primary is a calibrated mixture of seven "
        "models instead of one boosting model; and calibration on earlier out-of-fold races "
        f"replaces the single-race temperature {comparison['old_temperature']} (which "
        "sharpened every gap) with temperatures near 1.4 that flatten them.",
        "",
    ]
    movers = sorted(
        comparison["drivers"].items(),
        key=lambda item: -abs(item[1]["new"]["win"] - item[1]["old"]["win"]),
    )
    for driver, value in movers:
        change = value["new"]["win"] - value["old"]["win"]
        if abs(change) < 0.03:
            continue
        attribution = value["win_attribution"]
        stages = sorted(attribution.items(), key=lambda item: -abs(item[1]))
        main = stages[0][0].replace("_", " ")
        inputs = ", ".join(
            f"{label} {value['features'][key]:.2f}"
            for key, label in _INPUT_LABELS.items()
            if value["features"].get(key) is not None
        )
        lines.append(
            f"- **{driver}** {probability(value['old']['win'])} to "
            f"{probability(value['new']['win'])}: mostly {main} "
            f"({100 * attribution['models_and_features']:+.1f} pp models and features, "
            f"{100 * attribution['calibration']:+.1f} pp calibration, "
            f"{100 * attribution['unseen_circuit_rule']:+.1f} pp unseen-circuit rule). "
            f"Inputs: {inputs}."
        )
    wdc = comparison["wdc"]
    leader = max(wdc, key=lambda name: wdc[name]["old"] or 0)
    fixed = championship["uncertainty"]["sensitivity"]["fixed_strength_no_persistent_uncertainty"]
    lines += [
        "",
        f"- **{leader} title** {coarse(wdc[leader]['old'] or 0)} to "
        f"{coarse(wdc[leader]['new'])}: with strength fixed across events the new race "
        f"model alone gives {coarse(fixed['wdc'].get(leader, 0))}; the validated form drift "
        "between events accounts for the rest.",
    ]
    return lines


def render_report(
    manifest: dict[str, Any],
    race: dict[str, Any],
    championship: dict[str, Any],
    evaluation: dict[str, Any],
    protocol: dict[str, Any],
    comparison: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> str:
    event = manifest["event"]
    contract = manifest["cutoff_kind"]
    draws = race["draws"]
    primary = race["primary_model"]
    experimental = manifest["evaluation"]["experimental_model"]
    drivers = sorted(race["drivers"], key=lambda item: -item["winner_probability"])
    baselines = race["baselines"]
    stages = race["stages"]
    lines = [
        f"# Development prediction: {event['race_name']}",
        "",
        f"> **{manifest['warning']}**",
        "",
        "## Run",
        "",
    ]
    lines += _table(
        ["Field", "Value"],
        [
            ["Race", f"{event['race_name']} ({event['event_id']})"],
            [
                "Circuit",
                f"{event['circuit_name']} (`{event['circuit_id']}`, from the schedule "
                "circuit identifier, not the event title)",
            ],
            ["Race start (UTC)", event["race_start"]],
            ["Cutoff contract", f"`{contract}`"],
            ["prediction_timestamp_utc", manifest["prediction_timestamp_utc"]],
            ["First practice (UTC)", str(event["first_practice"])],
            ["Qualifying (UTC)", str(event["qualifying_start"])],
            ["Schedule observed at (UTC)", manifest["schedule_source"]["captured_at"]],
            ["Gold dataset", f"`{manifest['dataset']['gold']}`"],
            [
                "DNF dataset",
                f"`{manifest['dataset']['dnf']}` "
                f"({manifest['dataset']['dnf_known_labels']} audited labels)",
            ],
            ["Feature contract", manifest["feature_contract"]],
            [
                "Evaluation protocol",
                f"{manifest['protocol']['version']} (`{manifest['protocol']['sha256'][:12]}`)",
            ],
            ["Evidence tier", "training Gold, snapshot Development"],
            ["Model run ID", f"`{manifest['model_run_id']}`"],
            ["Primary development model", f"`{primary}` ({', '.join(race['primary_members'])})"],
            ["Baseline", "logistic winner/podium with Ridge finish order"],
            ["Experimental boosting model", f"`{experimental}`"],
            ["DNF model", f"`{manifest['evaluation']['dnf']['model']}`"],
            ["Roster source", manifest["roster_source"]],
            ["Execution device", ", ".join(manifest["execution"]["devices"])],
            ["Race draws, seed", f"{draws}, {race['seed']}"],
        ],
    )
    for note in manifest["notes"]:
        lines += ["", f"Note: {note}."]
    checks = manifest["validation_checks"]
    quality = manifest["sharpness"]["quality"]
    lines += [
        "",
        "## Checks",
        "",
        "Mathematical coherence and leakage checks: "
        + ("all passed." if all(checks.values()) else "FAILED."),
        "",
        f"Development quality (sharpness against the same model's historical out-of-fold "
        f"forecasts): **{quality['status']}**."
        + (" " + " ".join(quality["notes"]) if quality["notes"] else ""),
        "",
        "Coherent probabilities are not necessarily well calibrated. Calibration evidence is "
        "the historical out-of-fold section below.",
    ]
    ood = manifest["ood"]
    lines += ["", "## Out-of-distribution status", ""]
    lines.append(f"Status: **{ood['status']}**.")
    for reason in ood["reasons"]:
        lines.append(f"- {reason}")
    circuit = ood["circuit"]
    lines.append(
        f"- Circuit `{circuit['circuit_id']}` unseen in training: "
        f"{'yes' if circuit['unseen_in_training'] else 'no'} "
        f"({circuit['training_circuits']} training circuits). Circuit history features are "
        "left missing; no modern Sepang performance is invented."
    )
    lines.append(
        f"- Feature range violations: {len(ood['feature_range_violations'])}; novel "
        f"missing-value patterns: {len(ood['novel_missing_pattern_drivers'])}; drivers beyond "
        "the 99th percentile nearest-row distance: "
        f"{len(ood['distance']['beyond_99th_percentile'])}."
    )
    calibration_sources = {item["source"] for item in race["calibration"].values()}
    lines.append(
        "- Unseen-circuit calibration: "
        + (
            "the validated unseen-circuit rule changed at least one member's calibration."
            if "unseen_rule" in calibration_sources
            else "no member's unseen-circuit rule beat shared calibration out of fold, so "
            "shared calibration is used."
        )
    )
    lines += ["", "## Race probabilities", ""]
    lines.append(
        f"Small values keep their precision; `0 of {draws} draws` means the event never occurred "
        "in the joint draws. Logistic is the baseline; the experimental column is the best "
        "single boosting candidate."
    )
    lines.append("")
    lines += _table(
        [
            "Driver",
            "Team",
            "Win",
            "Podium",
            "DNF",
            "Expected",
            "80% range",
            "Logistic win",
            f"{experimental} win",
        ],
        [
            [
                item["driver_id"],
                item["constructor_id"],
                probability(item["winner_probability"], draws),
                probability(item["podium_probability"], draws),
                probability(item["dnf_model_probability"]),
                _num(item["expected_position"], 2),
                f"P{item['position_interval_80'][0]}-P{item['position_interval_80'][1]}",
                probability(baselines["logistic"][item["driver_id"]]["winner"]),
                probability(stages[f"candidate_{experimental}"][item["driver_id"]]["win"], draws),
            ]
            for item in drivers
        ],
    )
    lines += ["", "## Predicted finishing order", ""]
    lines.append(
        "By expected position. Sampled retirements trail finishers, so this is a modelled "
        "order, not an FIA classification."
    )
    lines.append("")
    for rank, item in enumerate(sorted(race["drivers"], key=lambda x: x["expected_position"]), 1):
        lines.append(
            f"{rank}. {item['driver_id']} ({item['constructor_id']}), expected "
            f"{item['expected_position']:.2f}, most likely P{item['most_likely_position']}"
        )
    dnf_eval = manifest["evaluation"]["dnf"]
    dnf_values = {round(item["dnf_model_probability"], 6) for item in race["drivers"]}
    lines += ["", "## DNF", ""]
    lines += _table(
        ["DNF model", "Audited labels", "Brier", "Log loss", "ECE", "Mean within-race SD"],
        [
            [
                name,
                str(values["labels"]),
                _num(values["brier_score"], 4),
                _num(values["log_loss"], 4),
                _num(values["ece"], 4),
                _num(values["mean_within_race_standard_deviation"], 4),
            ]
            for name, values in dnf_eval["candidates"].items()
        ],
    )
    lines.append("")
    if len(dnf_values) == 1:
        lines.append(
            f"**Limitation:** the selected DNF model is `{dnf_eval['model']}`. Driver-specific "
            "reliability features (driver and constructor audited DNF rates, circuit attrition) "
            "did not beat the audited base rate out of fold, so every driver receives the same "
            f"{probability(next(iter(dnf_values)))}. This is a field-wide rate, not an "
            "individualized prediction."
        )
    else:
        lines.append("DNF probabilities differ by driver through audited reliability features.")
    lines += ["", "## Historical evidence for this cutoff", ""]
    lines.append(
        f"Frozen protocol {manifest['protocol']['version']} on {evaluation['outer_races']} "
        "chronological outer Gold races, calibrated and selected only on earlier races, with "
        "this run's unavailable predictors hidden. Lower is better except top-1 accuracy."
    )
    lines.append("")
    models = evaluation["models"]
    shown = [primary, "selected_pipeline", *race["primary_members"], experimental]
    seen: list[str] = []
    for name in shown:
        if name in models and name not in seen:
            seen.append(name)
    lines += _table(
        [
            "Model",
            "Win LL",
            "Win Brier",
            "Top-1",
            "Podium Brier",
            "Finish MAE",
            "Win ECE",
            "Podium ECE",
            "Winners <1%",
            "Contenders",
        ],
        [
            *[_metric_row(name, models[name]["metrics"]) for name in seen],
            *[_metric_row(name, metrics) for name, metrics in evaluation["baselines"].items()],
        ],
    )
    paired = models[primary]["paired_uncertainty"]["logistic"]
    lines.append("")
    lines.append(
        f"Primary minus logistic, race-bootstrap 95% intervals: winner log loss "
        f"{paired['winner_log_loss']['mean_loss_delta']:+.4f} "
        f"[{paired['winner_log_loss']['bootstrap_95_percent_interval'][0]:+.4f}, "
        f"{paired['winner_log_loss']['bootstrap_95_percent_interval'][1]:+.4f}], podium Brier "
        f"{paired['podium_brier']['mean_loss_delta']:+.4f} "
        f"[{paired['podium_brier']['bootstrap_95_percent_interval'][0]:+.4f}, "
        f"{paired['podium_brier']['bootstrap_95_percent_interval'][1]:+.4f}], finish MAE "
        f"{paired['position_mae']['mean_loss_delta']:+.3f} "
        f"[{paired['position_mae']['bootstrap_95_percent_interval'][0]:+.3f}, "
        f"{paired['position_mae']['bootstrap_95_percent_interval'][1]:+.3f}]."
    )
    gate = models[primary]["formal_gate"]
    lines.append("")
    lines.append(
        "Formal task gate for the primary: "
        + "; ".join(f"{task} {value['status']} ({value['reason']})" for task, value in gate.items())
        + ". `no_selection` means the primary is a development choice, not a validated model."
    )
    protocol_primary = protocol["live"]["primary"]
    lines.append(
        f"Without this run's masking, the protocol evaluation selects `{protocol_primary}` "
        f"(`{manifest['evaluation']['protocol_evaluation']}`)."
    )
    lines += ["", "## Calibration diagnostics", ""]
    lines += _table(
        [
            "Member",
            "Temperature",
            "Prior shrinkage",
            "Source",
            "Uncalibrated win LL",
            "Calibrated win LL",
        ],
        [
            [
                name,
                str(values["temperature"]),
                str(values["shrink"]),
                values["source"],
                _num(models[name]["uncalibrated_metrics"]["winner"]["log_loss"]),
                _num(models[name]["metrics"]["winner"]["log_loss"]),
            ]
            for name, values in race["calibration"].items()
        ],
    )
    lines.append("")
    lines.append(
        "Temperature above 1 flattens a candidate's strengths; prior shrinkage mixes in a "
        "DNF-aware uniform race order. Both are fitted on earlier out-of-fold races only. "
        "Sigmoid and isotonic recalibration of single marginals were not adopted because they "
        "break the joint distribution's coherence; the ensemble mixture is the calibrated "
        "ensemble option."
    )
    curve = models[primary]["metrics"]["winner"]["reliability"]
    lines += ["", "Primary winner reliability curve (out of fold):", ""]
    lines += _table(
        ["Bin", "Driver rows", "Mean predicted", "Observed"],
        [
            [
                str(item["bin"]),
                str(item["count"]),
                _num(item["mean_predicted"], 4),
                _num(item["observed_rate"], 4),
            ]
            for item in curve
        ],
    )
    lines += ["", "## Sharpness", ""]
    lines += _table(
        ["Statistic", "Live", "Historical median", "Historical range", "Live percentile"],
        [
            [
                name,
                _num(value["live"]),
                _num(value["historical_median"]),
                f"{_num(value['historical_min'])} to {_num(value['historical_max'])}",
                f"{100 * value['historical_percentile']:.0f}%",
            ]
            for name, value in quality["metrics"].items()
        ],
    )
    drivers_section = manifest["feature_drivers"]
    lines += ["", "## Feature drivers", ""]
    lines.append("Global effects, not per-driver attributions.")
    for title, key, field in (
        (
            "Logistic strength, standardized coefficients",
            "logistic_pl_standardized_coefficients",
            "coefficient",
        ),
        (f"{experimental} split importance", f"{experimental}_importance_share", "share"),
    ):
        if key in drivers_section:
            lines += ["", title + ":", ""]
            lines += _table(
                ["Feature", field.capitalize()],
                [[item["feature"], _num(item[field])] for item in drivers_section[key]],
            )
    lines += ["", "## Missing features", ""]
    counts: dict[str, dict[str, int]] = {}
    for reasons in manifest["missing_feature_reasons"].values():
        for name, reason in reasons.items():
            counts.setdefault(name, {}).setdefault(reason, 0)
            counts[name][reason] += 1
    masked = set(manifest["evaluation"]["masked_for_training"])
    lines += _table(
        ["Feature", "Drivers missing", "Hidden in training", "Reasons"],
        [
            [
                name,
                str(sum(value.values())),
                "yes" if name in masked else "no",
                "; ".join(f"{reason} ({count})" for reason, count in sorted(value.items())),
            ]
            for name, value in sorted(counts.items())
        ],
    )
    lines.append("")
    lines.append(
        "Weather is not a predictor: no Gold training race has defensible point-in-time "
        "forecast evidence yet."
    )
    simulator = championship["simulator"]
    uncertainty = championship["uncertainty"]
    lines += ["", "## Development championship", ""]
    lines.append(
        f"**DEVELOPMENT ONLY.** {simulator['simulations']} simulated seasons over "
        f"{uncertainty['worlds']} persistent-strength worlds and "
        f"{len(championship['sessions'])} remaining sessions. Percentages are rounded to whole "
        "points because the model evidence does not support finer precision."
    )
    lines.append("")
    spread = uncertainty["bootstrap_parameter_spread"]
    persistent = uncertainty["persistent_strength"]
    drift = uncertainty["form_drift"]
    lines.append("Model uncertainty, separate from Monte Carlo race randomness:")
    lines.append("")
    lines.append(
        f"- Form drift: per-race driver strength random-walk variance "
        f"{uncertainty['per_race_drift_variance']} (on the calibrated utility scale), fitted on "
        f"{drift['pairs']} historical stale-form forecasts up to "
        f"{drift['protocol']['horizon_races']} races ahead"
        f"{' (at the edge of its grid)' if drift['grid_edge'] else ''}. Each simulated "
        "world draws one path, so a driver who drifts up stays up for the rest of that world."
    )
    lines.append(
        f"- Persistent offset: SD {uncertainty['persistent_standard_deviation']}. The observed "
        f"within-season residual correlation is {persistent['observed_icc']:.3f} "
        f"(95% interval {persistent['observed_icc_95_percent_interval'][0]:.3f} to "
        f"{persistent['observed_icc_95_percent_interval'][1]:.3f}) against "
        f"{persistent['simulated_icc_by_sigma'][0]['icc_mean']:.3f} expected with no persistent "
        "offset, so refreshed form features already carry persistence."
    )
    lines.append(
        f"- Parameter uncertainty: bootstrap refits of `{spread['model']}` move utilities by "
        f"{spread['mean_utility_standard_deviation']:.3f} SD on average. It is part of the "
        "out-of-fold errors behind both estimates above, so it is not added again."
    )
    lines.append("")
    lines += _table(
        ["Races ahead", "Historical forecasts", "Objective without drift", "With fitted drift"],
        [
            [
                lag,
                str(values["pairs"]),
                _num(values["objective_without_drift"]),
                _num(values["objective_with_selected_drift"]),
            ]
            for lag, values in drift["by_lag"].items()
        ],
    )
    lines.append("")
    lines.append(
        f"Monte Carlo: {simulator['simulations']} simulations are not independent estimates of "
        f"model truth. Title probabilities have Monte Carlo SE at most "
        f"{uncertainty['monte_carlo_standard_error_max']:.3f}, from the {uncertainty['worlds']} "
        "worlds."
    )
    for note in championship["standings_source"]["notes"]:
        lines.append(f"Starting points: {note}.")
    sensitivity = uncertainty["sensitivity"]
    for label, key in (("Drivers' championship", "wdc"), ("Constructors' championship", "wcc")):
        result = simulator[key]
        fixed = sensitivity["fixed_strength_no_persistent_uncertainty"][key]
        start = championship["starting_points"][
            "driver_points" if key == "wdc" else "constructor_points"
        ]
        ranked = sorted(
            result["mean_final_points"], key=lambda name: -result["mean_final_points"][name]
        )
        rows_out = []
        for name in ranked:
            places = result["final_position_probability"][name]
            if result["mean_final_points"][name] < 1 and start.get(name, 0) < 1:
                continue
            rows_out.append(
                [
                    name,
                    _num(start.get(name), 0),
                    _num(result["mean_final_points"][name], 0),
                    coarse(result["title_probability"][name]),
                    coarse(sum(places[:3])),
                    f"P{max(range(len(places)), key=lambda index: places[index]) + 1}",
                    coarse(fixed.get(name, 0)),
                ]
            )
        lines += ["", f"### {label}", ""]
        lines += _table(
            [
                "Name",
                "Points now",
                "Expected final",
                "Title",
                "Top 3",
                "Most likely",
                "Title if strength fixed",
            ],
            rows_out,
        )
        lines.append("")
        lines.append(
            f"Unresolved title tie probability: {coarse(result['unresolved_tie_probability'])}."
        )
    alternatives = {k: v for k, v in sensitivity.items() if k.startswith("single_candidate_")}
    if alternatives:
        leader = max(
            simulator["wdc"]["title_probability"], key=simulator["wdc"]["title_probability"].get
        )
        lines += [
            "",
            "Candidate-model sensitivity for the WDC leader "
            f"`{leader}`: "
            + ", ".join(
                name.removeprefix("single_candidate_") + " " + coarse(value["wdc"].get(leader, 0))
                for name, value in alternatives.items()
            )
            + ".",
        ]
    lines += ["", "Assumptions:", ""]
    lines.extend(f"- {item}" for item in championship["assumptions"])
    if comparison is not None:
        lines += ["", "## Comparison with the c52b674 forecast", ""]
        lines += _movement_narrative(comparison, championship)
        lines.append("")
        lines.append(
            f"The earlier run `{comparison['fixture_run_id']}` used a single `"
            f"{comparison['old_position_backend']}` model with temperature "
            f"{comparison['old_temperature']}, fitted on one calibration race, with qualifying "
            "and points features hidden. Win-probability changes are split into stages: new "
            "models and features before calibration, then calibration, then the unseen-circuit "
            "rule."
        )
        lines.append("")
        items = sorted(
            comparison["drivers"].items(),
            key=lambda item: -abs(item[1]["new"]["win"] - item[1]["old"]["win"]),
        )
        lines += _table(
            [
                "Driver",
                "Old win",
                "New win",
                "Models and features",
                "Calibration",
                "Unseen rule",
                "Old podium",
                "New podium",
                "Old exp.",
                "New exp.",
            ],
            [
                [
                    driver,
                    probability(value["old"]["win"]),
                    probability(value["new"]["win"]),
                    f"{100 * value['win_attribution']['models_and_features']:+.1f} pp",
                    f"{100 * value['win_attribution']['calibration']:+.1f} pp",
                    f"{100 * value['win_attribution']['unseen_circuit_rule']:+.1f} pp",
                    probability(value["old"]["podium"]),
                    probability(value["new"]["podium"]),
                    _num(value["old"]["expected_finish"], 2),
                    _num(value["new"]["expected_finish"], 2),
                ]
                for driver, value in items
            ],
        )
        lines += ["", "Championship movement:", ""]
        lines += _table(
            ["Name", "Old title", "New title"],
            [
                [name, coarse(value["old"] or 0), coarse(value["new"])]
                for name, value in sorted(
                    {**comparison["wdc"], **comparison["wcc"]}.items(),
                    key=lambda item: -(item[1]["old"] or 0),
                )
                if (value["old"] or 0) >= 0.001 or value["new"] >= 0.01
            ],
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m f1_ml_predictor predict-next-race",
        "```",
        "",
    ]
    return "\n".join(lines)
