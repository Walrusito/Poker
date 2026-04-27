import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from env.poker_env import PokerEnv
from models.policy_net import PolicyNet
from utils.information_set import InformationSetBuilder
from utils.run_comparison import collect_run_summaries, sort_run_summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Build a cross-play matrix between checkpointed runs")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints", help="Root directory containing checkpointed runs")
    parser.add_argument("--experiment", type=str, default="poker_cfr_ai", help="Experiment name to inspect")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt", choices=["best.pt", "latest.pt"], help="Checkpoint file to load from each run")
    parser.add_argument("--sort-key", type=str, default="ranking_value", help="Metric used to rank runs before selecting the top subset")
    parser.add_argument("--top-runs", type=int, default=4, help="How many runs to include when run-names is not provided")
    parser.add_argument("--run-names", type=str, default="", help="Optional comma-separated run names to include")
    parser.add_argument("--hands", type=int, default=120, help="Hands per matrix entry")
    parser.add_argument("--players", type=int, default=4, help="Number of players to simulate in each matchup")
    parser.add_argument("--starting-stack", type=int, default=10000, help="Starting stack in chips")
    parser.add_argument("--small-blind", type=int, default=50, help="Small blind in chips")
    parser.add_argument("--big-blind", type=int, default=100, help="Big blind in chips")
    parser.add_argument("--mc-simulations", type=int, default=150, help="Monte Carlo simulations for information-set equity estimates")
    parser.add_argument("--lut-simulations", type=int, default=1000, help="LUT simulations for preflop/flop equity tables")
    parser.add_argument("--lut-dir", type=str, default="data/lut", help="Directory for preflop/flop LUT files")
    parser.add_argument("--seed", type=int, default=7, help="Base seed for reproducible matchups")
    parser.add_argument("--output-csv", type=str, default="", help="Optional CSV output path")
    return parser.parse_args()


def select_runs(args):
    summaries = collect_run_summaries(args.checkpoint_dir, args.experiment)
    if args.run_names:
        target_names = {name.strip() for name in args.run_names.split(",") if name.strip()}
        summaries = [summary for summary in summaries if summary["run_name"] in target_names]
    else:
        summaries = sort_run_summaries(summaries, sort_key=args.sort_key)
        summaries = summaries[: max(1, args.top_runs)]

    selected = []
    for summary in summaries:
        if args.checkpoint_name == "best.pt":
            checkpoint_path = summary.get("best_policy_checkpoint") or summary["best_checkpoint"]
        else:
            checkpoint_path = summary.get("latest_policy_checkpoint") or summary["latest_checkpoint"]
        if not checkpoint_path or not Path(checkpoint_path).exists():
            continue
        selected.append({"run_name": summary["run_name"], "checkpoint_path": checkpoint_path})
    return selected


def load_policy(checkpoint_path, input_dim, output_dim, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PolicyNet.from_checkpoint_payload(
        payload,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
    )
    model.eval()
    return model


def masked_strategy(model, iss, env, state, player, actions, device):
    legal_actions = env.get_legal_actions()
    x = torch.tensor(iss.encode_vector(state, player), dtype=torch.float32, device=device)

    with torch.inference_mode():
        strategy = model(x.unsqueeze(0)).squeeze(0).cpu().numpy()

    mask = np.array([1.0 if action in legal_actions else 0.0 for action in actions], dtype=np.float32)
    strategy = strategy * mask
    total = float(np.sum(strategy))
    if total <= 0.0:
        strategy = mask / np.sum(mask)
    else:
        strategy = strategy / total
    return strategy


def evaluate_matchup(hero_model, villain_model, iss, args, device, seed_offset):
    env = PokerEnv(
        num_players=args.players,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        reward_unit="bb",
        seed=args.seed + seed_offset,
    )
    actions = tuple(env.ACTIONS)
    rng = np.random.default_rng(args.seed + seed_offset)
    total_hero_utility = 0.0

    for hand_idx in range(args.hands):
        hero_seat = hand_idx % env.num_players
        state = env.reset()
        done = False
        info = {}

        while not done:
            player = state["current_player"]
            model = hero_model if player == hero_seat else villain_model
            strategy = masked_strategy(model, iss, env, state, player, actions, device)
            action_idx = int(rng.choice(len(actions), p=strategy))
            state, _, done, info = env.step(actions[action_idx])

        total_hero_utility += float(info["terminal_utilities"][hero_seat])

    hero_ev = total_hero_utility / max(1, args.hands)
    return hero_ev * 100.0


def format_matrix(run_names, matrix):
    header = ["hero_vs_field"] + run_names
    lines = [" | ".join(header), " | ".join("-" * len(item) for item in header)]

    for hero_name in run_names:
        row = [hero_name]
        for villain_name in run_names:
            row.append(f"{matrix[(hero_name, villain_name)]:.2f}")
        lines.append(" | ".join(row))

    return "\n".join(lines)


def write_matrix_csv(run_names, matrix, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hero_vs_field", *run_names])
        for hero_name in run_names:
            writer.writerow([hero_name, *[f"{matrix[(hero_name, villain_name)]:.6f}" for villain_name in run_names]])


def main():
    args = parse_args()
    selected_runs = select_runs(args)
    if len(selected_runs) < 2:
        print("Need at least two runs to build a cross-play matrix.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_env = PokerEnv(
        num_players=args.players,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        reward_unit="bb",
        seed=args.seed,
    )
    iss = InformationSetBuilder(
        mc_simulations=args.mc_simulations,
        lut_simulations=args.lut_simulations,
        lut_dir=args.lut_dir,
        seed=args.seed,
    )

    loaded_models = {}
    for selected in selected_runs:
        loaded_models[selected["run_name"]] = load_policy(
            selected["checkpoint_path"],
            input_dim=iss.feature_dim,
            output_dim=len(probe_env.ACTIONS),
            device=device,
        )

    run_names = [selected["run_name"] for selected in selected_runs]
    matrix = {}
    matchup_tasks = []
    seed_offset = 0
    for hero_name in run_names:
        for villain_name in run_names:
            matchup_tasks.append((hero_name, villain_name, seed_offset))
            seed_offset += 97

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, len(matchup_tasks))) as executor:
        futures = {}
        for hero_name, villain_name, so in matchup_tasks:
            fut = executor.submit(
                evaluate_matchup,
                loaded_models[hero_name],
                loaded_models[villain_name],
                iss, args, device, seed_offset=so,
            )
            futures[fut] = (hero_name, villain_name)
        for fut in futures:
            hero_name, villain_name = futures[fut]
            matrix[(hero_name, villain_name)] = fut.result()

    print(format_matrix(run_names, matrix))

    if args.output_csv:
        write_matrix_csv(run_names, matrix, args.output_csv)
        print(f"\nCSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
