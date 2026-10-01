"""Print compact task metrics from one immutable chronological comparison."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if "stages" in report:
        for stage, result in report["stages"].items():
            print(stage)
            for backend, model in result["models"].items():
                metrics = model["metrics"]
                print(
                    " ",
                    backend,
                    "winner",
                    round(metrics["winner"]["log_loss"], 4),
                    "podium",
                    round(metrics["podium"]["log_loss"], 4),
                    "rank",
                    round(metrics["finishing_position"]["mean_absolute_error"], 4),
                )
                change = model["versus_previous_group"]
                if change:
                    print(
                        "   change versus prior",
                        {
                            key: (
                                round(value["mean_loss_delta"], 4),
                                [
                                    round(bound, 4)
                                    for bound in value["bootstrap_95_percent_interval"]
                                ],
                            )
                            for key, value in change.items()
                            if value.get("status") == "estimated"
                        },
                    )
        return
    print("run", report.get("run_metadata", {}).get("run_id"))
    print("dataset", report["benchmark_manifest_sha256"])
    print("eligible", report["observation_counts"]["eligible_events_by_cutoff"])
    print("task_selection", report.get("task_selection"))
    for cohort, backends in report["comparisons"].items():
        print("cohort", cohort)
        for backend, result in backends.items():
            metrics = result["metrics"]
            print(
                backend,
                "paired",
                result["paired_cohorts"],
                "winner",
                metrics["winner"].get("log_loss"),
                "podium",
                metrics["podium"].get("log_loss"),
                "dnf",
                metrics["dnf"].get("log_loss"),
                "rank_mae",
                metrics["finishing_position"].get("mean_absolute_error"),
            )
            print("  regressions", result["regressions"])
            for baseline, tasks in result["paired_uncertainty"].items():
                print(
                    "  versus",
                    baseline,
                    {
                        task: (
                            value.get("mean_loss_delta"),
                            value.get("bootstrap_95_percent_interval"),
                        )
                        for task, value in tasks.items()
                    },
                )
            if backend == next(iter(backends)):
                print(
                    "  baselines",
                    {
                        name: {
                            task: values.get("log_loss", values.get("mean_absolute_error"))
                            for task, values in baseline_metrics.items()
                        }
                        for name, baseline_metrics in result["baselines"].items()
                    },
                )


if __name__ == "__main__":
    main()
