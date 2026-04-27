import argparse

from utils.run_comparison import collect_run_summaries, format_run_table, sort_run_summaries, write_run_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Compare checkpointed poker training runs")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints", help="Root directory containing checkpointed runs")
    parser.add_argument("--experiment", type=str, default="poker_cfr_ai", help="Experiment name to inspect")
    parser.add_argument(
        "--sort-key",
        type=str,
        default="best_vs_random_bb_per_100",
        choices=[
            "best_vs_random_bb_per_100",
            "best_vs_population_bb_per_100",
            "latest_vs_random_bb_per_100",
            "latest_vs_population_bb_per_100",
            "best_vs_snapshot_bb_per_100",
            "latest_vs_snapshot_bb_per_100",
            "best_vs_heuristic_bb_per_100",
            "latest_vs_heuristic_bb_per_100",
            "ranking_value",
        ],
        help="Metric used to rank runs",
    )
    parser.add_argument("--output-csv", type=str, default="", help="Optional CSV output path")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = collect_run_summaries(args.checkpoint_dir, args.experiment)
    summaries = sort_run_summaries(summaries, sort_key=args.sort_key)

    if not summaries:
        print("No runs found.")
        return

    print(format_run_table(summaries))

    if args.output_csv:
        write_run_csv(summaries, args.output_csv)
        print(f"\nCSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
