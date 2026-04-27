import csv
import json
from pathlib import Path


def _rank_value(summary):
    for key in (
        "best_vs_population_bb_per_100",
        "best_vs_snapshot_bb_per_100",
        "best_vs_random_bb_per_100",
    ):
        value = summary.get(key)
        if value is not None:
            return value
    return None


def collect_run_summaries(root_dir: str, experiment: str):
    experiment_dir = Path(root_dir) / experiment
    if not experiment_dir.exists():
        return []

    summaries = []
    for summary_path in experiment_dir.glob("*/run_summary.json"):
        with summary_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        latest_metrics = {}
        checkpoints = payload.get("checkpoints", [])
        if checkpoints:
            latest_metrics = checkpoints[-1].get("metrics", {})

        run_dir = summary_path.parent
        summaries.append(
            {
                "run_name": payload.get("run_name"),
                "seed": payload.get("seed"),
                "latest_iteration": payload.get("latest_iteration"),
                "best_iteration": payload.get("best_iteration"),
                "best_vs_random_bb_per_100": payload.get("best_vs_random_bb_per_100"),
                "best_vs_snapshot_bb_per_100": payload.get("best_vs_snapshot_bb_per_100"),
                "best_vs_population_bb_per_100": payload.get("best_vs_population_bb_per_100"),
                "best_vs_heuristic_bb_per_100": payload.get("best_vs_heuristic_bb_per_100"),
                "latest_vs_random_bb_per_100": latest_metrics.get("vs_random_bb_per_100"),
                "latest_vs_snapshot_bb_per_100": latest_metrics.get("vs_snapshot_bb_per_100"),
                "latest_vs_population_bb_per_100": latest_metrics.get("vs_population_bb_per_100"),
                "latest_vs_heuristic_bb_per_100": latest_metrics.get("vs_heuristic_bb_per_100"),
                "latest_self_play_seat_ev_std": latest_metrics.get("self_play_seat_ev_std"),
                "run_dir": str(run_dir),
                "best_checkpoint": str(run_dir / "best.pt"),
                "latest_checkpoint": str(run_dir / "latest.pt"),
                "ranking_value": _rank_value(payload),
            }
        )

    return summaries


def collect_population_checkpoints(
    root_dir: str,
    experiment: str,
    checkpoint_name: str = "best.pt",
    exclude_run_dir: str = None,
    limit: int = None,
):
    summaries = collect_run_summaries(root_dir, experiment)
    summaries = sort_run_summaries(summaries, sort_key="ranking_value")

    checkpoints = []
    for summary in summaries:
        if exclude_run_dir is not None and Path(summary["run_dir"]) == Path(exclude_run_dir):
            continue

        checkpoint_path = Path(summary["run_dir"]) / checkpoint_name
        if checkpoint_path.exists():
            checkpoints.append(str(checkpoint_path))

    if limit is not None:
        checkpoints = checkpoints[:limit]

    return checkpoints


def sort_run_summaries(summaries, sort_key="best_vs_random_bb_per_100"):
    return sorted(
        summaries,
        key=lambda item: (
            item.get(sort_key) is not None,
            item.get(sort_key) if item.get(sort_key) is not None else float("-inf"),
        ),
        reverse=True,
    )


def format_run_table(summaries):
    headers = [
        "run_name",
        "seed",
        "latest_iteration",
        "best_iteration",
        "best_vs_random_bb_per_100",
        "best_vs_snapshot_bb_per_100",
        "best_vs_population_bb_per_100",
        "latest_vs_population_bb_per_100",
    ]

    lines = [" | ".join(headers), " | ".join("-" * len(header) for header in headers)]
    for item in summaries:
        row = []
        for header in headers:
            value = item.get(header)
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        lines.append(" | ".join(row))
    return "\n".join(lines)


def write_run_csv(summaries, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys()) if summaries else []
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(item)
