"""Readable markdown for one development prediction run."""

from __future__ import annotations

from typing import Any


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _metric_rows(comparison: dict[str, Any], backend: str) -> list[list[str]]:
    keys = (
        ("winner", "log_loss"),
        ("podium", "brier_score"),
        ("finishing_position", "mean_absolute_error"),
    )
    rows = []
    sources = {backend: comparison["metrics"], **comparison["baselines"]}
    for name, metrics in sources.items():
        rows.append([name, *(_num(metrics[task].get(metric)) for task, metric in keys)])
    return rows


def render_report(
    manifest: dict[str, Any],
    race: dict[str, Any],
    championship: dict[str, Any],
    table_rows: list[dict[str, Any]],
) -> str:
    event = manifest["event"]
    cutoff = manifest["prediction_cutoff"]
    kind = cutoff["kind"]
    model = manifest["models"][kind]
    choice = manifest["model_choice"]
    position_backend = choice["position"]["backend"]
    by_driver = {row["driver_id"]: row for row in table_rows}
    lines = [
        f"# Development prediction: {event['race_name']}",
        "",
        f"> **{manifest['warning']}**",
        "",
        "## Run",
        "",
    ]
    snapshot = manifest["snapshot"][kind]
    lines += _table(
        ["Field", "Value"],
        [
            ["Race", f"{event['race_name']} ({event['event_id']})"],
            ["Race start (UTC)", event["race_start"]],
            ["Prediction cutoff", f"`{kind}` ({cutoff['reason']})"],
            ["prediction_timestamp_utc", cutoff["prediction_timestamp_utc"]],
            ["Qualifying start (UTC)", str(event["qualifying_start"])],
            ["Schedule observed at (UTC)", manifest["schedule_source"]["captured_at"]],
            ["Snapshot", f"`{snapshot['path']}`"],
            ["Snapshot SHA-256", f"`{snapshot['sha256']}`"],
            ["Dataset version", f"`{manifest['dataset']['dataset_version']}`"],
            [
                "Gold coverage",
                f"{manifest['dataset']['gold_races']} races, "
                f"{manifest['dataset']['gold_rows']} rows",
            ],
            ["DNF dataset version", f"`{manifest['dnf_dataset']['dataset_version']}`"],
            ["Feature schema version", manifest["feature_schema_version"]],
            [
                "Evidence tier",
                f"training {manifest['evidence_tier']['training']}, snapshot "
                f"{manifest['evidence_tier']['prediction_snapshot']}",
            ],
            ["Model run ID", f"`{manifest['model_run_id']}`"],
            ["Model version", manifest["model_version"]],
            [
                "Position model",
                f"{position_backend} ({choice['position']['status']})",
            ],
            ["DNF model", f"{choice['dnf']['model']} ({choice['dnf']['status']})"],
            [
                "Reference runs",
                f"position `{manifest['reference_runs']['position']['run_id']}`, "
                f"DNF `{manifest['reference_runs']['dnf']['run_id']}`",
            ],
            ["Evaluation protocol", manifest["evaluation_protocol"]["version"]],
            ["Roster source", manifest["roster_source"]],
            ["Execution device", ", ".join(sorted(set(manifest["execution"]["devices"].values())))],
            ["Race draws, seed", f"{manifest['race_draws']}, {manifest['seeds']['race']}"],
        ],
    )
    drivers = sorted(race["drivers"], key=lambda row: -row["winner_probability"])
    lines += ["", "## Win and podium probabilities", ""]
    lines += _table(
        ["Driver", "Team", "Win", "Podium", "DNF", "Expected finish", "Most likely", "80% range"],
        [
            [
                row["driver_id"],
                by_driver[row["driver_id"]]["constructor_id"],
                _pct(row["winner_probability"]),
                _pct(row["podium_probability"]),
                _pct(row["dnf_probability"]),
                _num(row["expected_position"], 2),
                f"P{row['most_likely_position']}",
                f"P{row['position_interval_80'][0]} to P{row['position_interval_80'][1]}",
            ]
            for row in drivers
        ],
    )
    lines += ["", "## Predicted finishing order", ""]
    lines.append(
        "Ordered by expected finishing position from the joint distribution. Retirements "
        "sampled by the DNF model trail finishers, so this is a modelled order, not an FIA "
        "classification."
    )
    lines.append("")
    for rank, row in enumerate(
        sorted(race["drivers"], key=lambda item: item["expected_position"]), 1
    ):
        lines.append(
            f"{rank}. {row['driver_id']} ({by_driver[row['driver_id']]['constructor_id']}), "
            f"expected {row['expected_position']:.2f}"
        )
    lines += ["", "## DNF risk ranking", ""]
    dnf_meta = model["metadata"]["dnf_model"]
    lines.append(
        f"DNF comes from a separate `{choice['dnf']['model']}` model fitted on "
        f"{dnf_meta['training_labels']} audited binary DNF labels "
        f"({dnf_meta['training_retirements']} retirements), not from pace."
    )
    masked_dnf = (model.get("availability_evaluation") or {}).get("dnf_masked_features", [])
    if masked_dnf:
        lines.append(
            f"At this cutoff {len(masked_dnf)} of its predictors are unavailable, so drivers "
            "with identical remaining inputs share the same estimate."
        )
    lines.append(
        "The model column is the DNF model's probability. The sampled column is its frequency "
        "in the joint race draws, which carries Monte Carlo noise and is what the finish "
        "distribution uses."
    )
    lines.append("")
    lines += _table(
        ["Driver", "DNF (model)", "DNF (sampled)"],
        [
            [
                row["driver_id"],
                _pct(row["dnf_model_probability"]),
                _pct(row["dnf_probability"]),
            ]
            for row in sorted(
                race["drivers"],
                key=lambda item: (-item["dnf_model_probability"], item["driver_id"]),
            )
        ],
    )
    lines += ["", "## Baseline comparison", ""]
    lines.append(
        "Existing logistic and Ridge baselines fitted on the same cutoff-matched Gold history. "
        "The logistic baseline has the lowest observed historical loss on winner, podium and "
        "finish tasks, which is one reason no model is validated."
    )
    lines.append("")
    lines += _table(
        [
            "Driver",
            "Model win",
            "Logistic win",
            "Model podium",
            "Logistic podium",
            "Model rank",
            "Ridge rank",
        ],
        [
            [
                row["driver_id"],
                _pct(by_driver[row["driver_id"]]["win_probability"]),
                _pct(by_driver[row["driver_id"]]["logistic_win_probability"]),
                _pct(by_driver[row["driver_id"]]["podium_probability"]),
                _pct(by_driver[row["driver_id"]]["logistic_podium_probability"]),
                str(rank),
                str(by_driver[row["driver_id"]]["linear_finish_rank"]),
            ]
            for rank, row in enumerate(
                sorted(race["drivers"], key=lambda item: item["expected_position"]), 1
            )
        ],
    )
    lines += ["", "## Model uncertainty", ""]
    lines.append(
        f"Monte Carlo standard error per probability is at most "
        f"{race['simulation_standard_error_max']:.4f} ({race['draws']} draws). Race-order "
        f"temperature is {race['temperature']}, chosen from a fixed grid on one calibration "
        "race under the frozen protocol, which can make probabilities overconfident. "
        "Sampling error is not model error."
    )
    lines.append("")
    lines.append(
        "Historical outer-fold metrics on the frozen post-qualifying run "
        f"(`{manifest['reference_runs']['position']['run_id']}`, 92 paired races, lower is better):"
    )
    lines.append("")
    position_metrics = choice["position"]
    lines += _table(
        ["Model", "Winner log loss", "Podium Brier", "Finish MAE"],
        [
            [name, *(_num(values[task]) for task in ("winner", "podium", "finishing_position"))]
            for name, values in {
                **position_metrics["candidate_metrics"],
                **position_metrics["baseline_metrics"],
            }.items()
        ],
    )
    evaluation = model.get("availability_evaluation")
    comparisons: dict[str, Any] = (
        next(iter(evaluation["position"].values()), {}) if evaluation else {}
    )
    if evaluation and position_backend in comparisons:
        comparison = comparisons[position_backend]
        lines += [
            "",
            "Cutoff-matched diagnostic evaluation, same frozen folds with the unavailable "
            f"predictors hidden ({len(comparison['paired_event_ids'])} paired races, "
            "diagnostic only):",
            "",
        ]
        lines += _table(
            ["Model", "Winner log loss", "Podium Brier", "Finish MAE"],
            _metric_rows(comparison, position_backend),
        )
        interval = comparison["paired_uncertainty"]["logistic"].get("winner_log_loss", {})
        bounds = interval.get("bootstrap_95_percent_interval")
        if bounds:
            lines.append("")
            lines.append(
                f"Winner log loss minus logistic: {interval['mean_loss_delta']:.4f}, "
                f"race-bootstrap 95% interval {bounds[0]:.4f} to {bounds[1]:.4f}."
            )
        dnf_baselines = evaluation.get("dnf_baselines", {})
        if dnf_baselines.get("logistic", {}).get("dnf"):
            dnf_metrics = dnf_baselines["logistic"]["dnf"]
            lines.append("")
            lines.append(
                "Cutoff-matched logistic DNF Brier score "
                f"{_num(dnf_metrics.get('brier_score'), 4)} "
                f"on {evaluation.get('dnf_evaluated_events')} audited DNF races."
            )
    lines += [
        "",
        "The scoring Gold version has no binary DNF labels, so both position evaluations "
        "compose the position model with the frozen Beta(1,1) DNF prior. They describe the "
        "position model and its baselines, not this run's composed model with the audited "
        "DNF model.",
    ]
    lines += ["", "## Feature drivers", ""]
    importance = model["feature_importance"]
    if importance:
        lines.append(
            f"Global split importance of the {position_backend} position model. This is not a "
            "per-driver attribution."
        )
        lines.append("")
        lines += _table(
            ["Feature", "Share"], [[item["feature"], _pct(item["share"])] for item in importance]
        )
    else:
        lines.append("The selected position backend does not expose feature importance.")
    lines += ["", "## Missing features", ""]
    missing = manifest["missing_features"][kind]
    absent = [
        item
        for item in missing
        if item["live_present"] < item["live_rows"] and item["training_present"]
    ]
    lines.append(
        "Trained predictors that are missing for at least one driver. Predictors missing for "
        "every driver are hidden from training as well, so they are never median-filled."
    )
    lines.append("")
    lines += _table(
        ["Feature", "Live rows", "Training rows", "Hidden in training", "Reason"],
        [
            [
                item["feature"],
                f"{item['live_present']}/{item['live_rows']}",
                str(item["training_present"]),
                "yes" if item["masked_for_training"] else "no",
                "; ".join(item["reasons"]) or "n/a",
            ]
            for item in absent
        ],
    )
    if model["scoring_gate"]:
        lines += [
            "",
            "Championship point features fail the strict audited ledger gate: "
            + "; ".join(model["scoring_gate"])
            + ".",
        ]
    untrained = [item["feature"] for item in missing if not item["training_present"]]
    if untrained:
        lines += [
            "",
            f"{len(untrained)} predictors have no Gold training coverage and stay missing: "
            + ", ".join(untrained)
            + ".",
        ]
    gate = manifest["snapshot"][kind]
    lines += ["", "## Development championship", ""]
    simulator = championship["simulator"]
    lines.append(
        f"**DEVELOPMENT ONLY.** {simulator['simulations']} seeded simulations "
        f"(seed {simulator['seed']}) of {len(championship['sessions'])} remaining sessions using "
        "only this run's development race samples. These are not validated forecasts."
    )
    for note in championship["standings_source"]["notes"]:
        lines.append(f"Starting points: {note}.")
    lines.append("")
    for label, key in (("Drivers' championship", "wdc"), ("Constructors' championship", "wcc")):
        result = simulator[key]
        start = championship["starting_points"][
            "driver_points" if key == "wdc" else "constructor_points"
        ]
        ranked = sorted(
            result["mean_final_points"], key=lambda name: -result["mean_final_points"][name]
        )
        rows = []
        for name in ranked:
            places = result["final_position_probability"][name]
            rows.append(
                [
                    name,
                    _num(start.get(name), 0),
                    _num(result["mean_final_points"][name], 1),
                    _pct(result["title_probability"][name]),
                    _pct(sum(places[:3])),
                    f"P{max(range(len(places)), key=lambda index: places[index]) + 1}",
                ]
            )
        lines += [f"### {label}", ""]
        lines += _table(
            ["Name", "Points now", "Expected final points", "Title", "Top 3", "Most likely final"],
            rows,
        )
        lines.append("")
        lines.append(
            f"Unresolved title tie probability: {_pct(result['unresolved_tie_probability'])}."
        )
        lines.append("")
    lines += ["Assumptions:", ""]
    lines.extend(f"- {item}" for item in championship["assumptions"])
    lines += ["", "## Validation checks", ""]
    lines.extend(
        f"- {name}: passed" for name, value in manifest["validation_checks"].items() if value
    )
    lines += [
        "",
        "## Reproduce",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m f1_ml_predictor predict-next-race",
        "```",
        "",
        f"Snapshot manifest: `{gate['manifest_path']}`.",
        "",
    ]
    return "\n".join(lines)
