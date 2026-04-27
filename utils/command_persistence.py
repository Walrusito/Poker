from pathlib import Path


def persist_run_command(args, command_file: str):
    parts = [
        "docker compose run --rm poker-ai",
        f"--iterations {args.iterations}",
        f"--episodes {args.episodes}",
        f"--eval-hands {args.eval_hands}",
        f"--eval-random-hands {args.eval_random_hands}",
        f"--eval-snapshot-hands {args.eval_snapshot_hands}",
        f"--eval-population-hands {args.eval_population_hands}",
        f"--eval-heuristic-hands {args.eval_heuristic_hands}",
        f"--players {args.players}",
        f"--starting-stack {args.starting_stack}",
        f"--small-blind {args.small_blind}",
        f"--big-blind {args.big_blind}",
        f"--mc-simulations {args.mc_simulations}",
        f"--lut-simulations {args.lut_simulations}",
        f"--lut-dir {args.lut_dir}",
        f"--rollouts-per-action {args.rollouts_per_action}",
        f"--feature-cache-size {args.feature_cache_size}",
        f"--batch-size {args.batch_size}",
        f"--regret-epochs {args.regret_epochs}",
        f"--policy-epochs {args.policy_epochs}",
        f"--gradient-clip {args.gradient_clip}",
        f"--parallel-workers {args.parallel_workers}",
        f"--snapshot-pool-size {args.snapshot_pool_size}",
        f"--max-snapshot-cache {args.max_snapshot_cache}",
        f"--population-run-limit {args.population_run_limit}",
        f"--population-checkpoint-name {args.population_checkpoint_name}",
        f"--population-mix-prob {args.population_mix_prob}",
        f"--checkpoint-interval {args.checkpoint_interval}",
        f"--early-stop-patience {getattr(args, 'early_stop_patience', 0)}",
        f"--early-stop-min-iters {getattr(args, 'early_stop_min_iters', 5)}",
        f"--early-stop-entropy-floor {getattr(args, 'early_stop_entropy_floor', 0.02)}",
        f"--early-stop-entropy-patience {getattr(args, 'early_stop_entropy_patience', 3)}",
        f"--early-stop-regret-loss-ceiling {getattr(args, 'early_stop_regret_loss_ceiling', 500.0)}",
        f"--early-stop-policy-loss-ceiling {getattr(args, 'early_stop_policy_loss_ceiling', 25.0)}",
        f"--checkpoint-dir {args.checkpoint_dir}",
        f"--resume-mode {args.resume_mode}",
        f"--seed {args.seed}",
        f"--experiment {args.experiment}",
        f"--command-file {args.command_file}",
    ]

    if getattr(args, "deterministic", False):
        parts.append("--deterministic")

    if getattr(args, "run_name", ""):
        parts.append(f"--run-name {args.run_name}")

    command = " ".join(parts)
    path = Path(command_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(command + "\n", encoding="utf-8")
    return command
