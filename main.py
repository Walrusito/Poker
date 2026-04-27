import argparse
import json
import random

import numpy as np
import torch

from env.deck import Deck
from env.poker_env import PokerEnv
from train.train_deep_cfr import DeepCFRTrainer
from utils.checkpointing import CheckpointManager
from utils.command_persistence import persist_run_command
from utils.logging import end_experiment, log_artifact, log_param, set_run_tags, start_experiment_run


def parse_args():
    parser = argparse.ArgumentParser(description="Poker CFR AI Training")
    parser.add_argument("--iterations", type=int, default=100, help="Deep CFR training iterations")
    parser.add_argument("--episodes", type=int, default=2000, help="Self-play episodes per iteration (was 250; needs to be much higher for meaningful regret signal)")
    parser.add_argument("--eval-hands", type=int, default=5000, help="Hands to evaluate in self-play after each iteration (was 500; more = lower variance)")
    parser.add_argument("--eval-random-hands", type=int, default=0, help="Hands to evaluate against random opponents after each iteration (0 = auto)")
    parser.add_argument("--eval-snapshot-hands", type=int, default=0, help="Hands to evaluate against historical snapshot opponents after each iteration (0 = auto)")
    parser.add_argument("--eval-population-hands", type=int, default=0, help="Hands to evaluate against a mixed population of external historical runs after each iteration (0 = auto)")
    parser.add_argument("--eval-heuristic-hands", type=int, default=0, help="Hands to evaluate against a heuristic opponent baseline after each iteration (0 = auto)")
    parser.add_argument("--players", type=int, default=2, help="Number of players to simulate (2-9)")
    parser.add_argument("--starting-stack", type=int, default=10000, help="Starting stack in chips")
    parser.add_argument("--small-blind", type=int, default=50, help="Small blind in chips")
    parser.add_argument("--big-blind", type=int, default=100, help="Big blind in chips")
    parser.add_argument("--mc-simulations", type=int, default=200, help="Monte Carlo simulations for non-LUT equity")
    parser.add_argument("--lut-simulations", type=int, default=1500, help="Simulations used to populate preflop/flop LUTs")
    parser.add_argument("--lut-dir", type=str, default="data/lut", help="Directory for preflop/flop LUT files")
    parser.add_argument(
        "--street-bet-multipliers",
        type=str,
        default="",
        help="JSON override for open bet multipliers by street, e.g. {\"flop\":{\"bet_33\":0.33}}",
    )
    parser.add_argument(
        "--street-raise-multipliers",
        type=str,
        default="",
        help="JSON override for raise multipliers by street, e.g. {\"turn\":{\"bet_150\":1.5}}",
    )
    parser.add_argument("--rollouts-per-action", type=int, default=1, help="Shared-policy rollout samples used to estimate unsampled actions")
    parser.add_argument("--feature-cache-size", type=int, default=50000, help="Information-set feature cache size")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for regret/policy updates")
    parser.add_argument("--regret-epochs", type=int, default=4, help="Gradient epochs for the regret network per iteration (was 2; more epochs = stronger regret signal)")
    parser.add_argument("--policy-epochs", type=int, default=1, help="Gradient epochs for the policy network per iteration (was 2; fewer epochs prevents entropy collapse)")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel workers for self-play episode collection and rollout evaluation")
    parser.add_argument("--disable-deterministic-parallel", action="store_true", help="Allow non-deterministic merge/order in parallel workers")
    parser.add_argument("--dataloader-workers", type=int, default=2, help="DataLoader workers for policy/regret training")
    parser.add_argument("--torch-num-threads", type=int, default=16, help="CPU threads used by Torch ops")
    parser.add_argument("--require-cuda", action="store_true", help="Fail fast if CUDA is not available")
    parser.add_argument("--rollout-batch-size", type=int, default=32, help="Batch size for policy rollouts in action utility estimation")
    parser.add_argument("--use-torch-equity", action="store_true", help="Use torch backend for Monte Carlo equity sampling")
    parser.add_argument("--torch-equity-device", type=str, default="cuda", help="Torch device for equity sampling backend")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision (CUDA only)")
    parser.add_argument("--use-torch-compile", action="store_true", help="Enable torch.compile for nets (PyTorch 2.x)")
    parser.add_argument("--disable-reach-weighting", action="store_true", help="Disable CFR-style reach/counterfactual sample weighting")
    parser.add_argument("--reach-weight-clip", type=float, default=100.0, help="Upper bound for reach-based sample weights")
    parser.add_argument("--reach-weight-mode", type=str, default="linear", choices=["linear", "sqrt"], help="Transform for CFR reach weights")
    parser.add_argument("--disable-reach-auto-clip", action="store_true", help="Disable automatic quantile-based reach-weight clipping")
    parser.add_argument("--reach-auto-clip-quantile", type=float, default=0.95, help="Quantile used to estimate effective reach-weight clip")
    parser.add_argument("--snapshot-pool-size", type=int, default=4, help="How many historical checkpoints to use when evaluating against snapshot opponents")
    parser.add_argument("--max-snapshot-cache", type=int, default=8, help="How many snapshot policies to cache in memory")
    parser.add_argument("--population-run-limit", type=int, default=6, help="How many external runs to include in the population opponent pool")
    parser.add_argument("--population-checkpoint-name", type=str, default="best.pt", choices=["best.pt", "latest.pt"], help="Which checkpoint file to pull from external runs")
    parser.add_argument("--population-mix-prob", type=float, default=0.0, help="Probability of using a population policy inside rollout simulations")
    parser.add_argument("--policy-smoothing-alpha", type=float, default=0.05, help="Uniform legal-action mixture weight for policy smoothing")
    parser.add_argument("--entropy-regularization", type=float, default=0.03, help="Entropy regularization weight in policy training (0.01-0.05; higher = more exploration; prevents entropy collapse)")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save a full checkpoint every N completed iterations")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop if robust score does not improve for N iterations (0 disables)")
    parser.add_argument("--early-stop-min-iters", type=int, default=5, help="Minimum completed iterations before early-stop checks")
    parser.add_argument("--early-stop-entropy-floor", type=float, default=0.02, help="Trigger stop if avg policy entropy stays below this floor")
    parser.add_argument("--early-stop-entropy-patience", type=int, default=3, help="How many consecutive low-entropy iterations trigger stop")
    parser.add_argument("--early-stop-regret-loss-ceiling", type=float, default=500.0, help="Trigger stop if regret loss exceeds this ceiling")
    parser.add_argument("--early-stop-policy-loss-ceiling", type=float, default=25.0, help="Trigger stop if policy loss exceeds this ceiling")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints", help="Root directory for saved checkpoints")
    parser.add_argument("--run-name", type=str, default="", help="Optional checkpoint run name")
    parser.add_argument("--resume-mode", type=str, default="auto", choices=["auto", "never"], help="Auto-resume the latest checkpoint if one exists")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable stricter deterministic behavior (PyTorch deterministic algorithms + stable CuDNN settings)",
    )
    parser.add_argument("--checkpoint-keep-last", type=int, default=0, help="Keep only the N most recent iter_XXXX.pt files (0 = keep all)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers for RegretNet and PolicyNet")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for RegretNet and PolicyNet")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for RegretNet and PolicyNet (0.0 = disabled)")
    parser.add_argument("--experiment", type=str, default="poker_cfr_ai", help="MLflow experiment name")
    parser.add_argument("--command-file", type=str, default="artifacts/last_run_command.txt", help="File used to persist the Docker command")
    return parser.parse_args()


def parse_street_multipliers(raw_value: str):
    if not raw_value:
        return None
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("street multipliers must be a JSON object")
    return parsed


def set_global_seed(seed, deterministic: bool = False):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "cudnn"):
        # Faster autotuning on fixed tensor shapes.
        torch.backends.cudnn.benchmark = True


def configure_torch_runtime(torch_num_threads: int, require_cuda: bool = False):
    if torch_num_threads and torch_num_threads > 0:
        torch.set_num_threads(int(torch_num_threads))
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but not available. Check Docker GPU runtime and torch CUDA wheel.")
    print(f"[Runtime] torch={torch.__version__} cuda_available={cuda_available} device_count={torch.cuda.device_count()}")
    if cuda_available:
        print(f"[Runtime] cuda_version={torch.version.cuda} device={torch.cuda.get_device_name(0)}")


def build_mlflow_params(args, run_context):
    return {
        "players": args.players,
        "starting_stack": args.starting_stack,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
        "mc_simulations": args.mc_simulations,
        "lut_simulations": args.lut_simulations,
        "street_bet_multipliers": args.street_bet_multipliers,
        "street_raise_multipliers": args.street_raise_multipliers,
        "rollouts_per_action": args.rollouts_per_action,
        "batch_size": args.batch_size,
        "regret_epochs": args.regret_epochs,
        "policy_epochs": args.policy_epochs,
        "gradient_clip": args.gradient_clip,
        "snapshot_pool_size": args.snapshot_pool_size,
        "population_run_limit": args.population_run_limit,
        "population_checkpoint_name": args.population_checkpoint_name,
        "population_mix_prob": args.population_mix_prob,
        "policy_smoothing_alpha": args.policy_smoothing_alpha,
        "entropy_regularization": args.entropy_regularization,
        "seed": args.seed,
        "deterministic": int(args.deterministic),
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "use_reach_weighting": int(not args.disable_reach_weighting),
        "reach_weight_mode": args.reach_weight_mode,
        "reach_weight_clip": args.reach_weight_clip,
        "reach_weight_auto_clip": int(not args.disable_reach_auto_clip),
        "reach_weight_auto_quantile": args.reach_auto_clip_quantile,
        "run_name": run_context["run_name"],
    }


def build_mlflow_tags(args, run_context, command):
    return {
        "poker.resume_mode": args.resume_mode,
        "poker.resumed_from_checkpoint": int(run_context["is_resumed"]),
        "poker.run_dir": run_context["run_dir"],
        "poker.command_file": args.command_file,
        "poker.command": command,
        "poker.iterations_target": args.iterations,
        "poker.episodes_per_iteration": args.episodes,
        "poker.eval_hands": args.eval_hands,
        "poker.eval_random_hands": args.eval_random_hands,
        "poker.eval_snapshot_hands": args.eval_snapshot_hands,
        "poker.eval_population_hands": args.eval_population_hands,
        "poker.eval_heuristic_hands": args.eval_heuristic_hands,
        "poker.lut_dir": args.lut_dir,
        "poker.feature_cache_size": args.feature_cache_size,
        "poker.parallel_workers": args.parallel_workers,
        "poker.deterministic_parallel": int(not args.disable_deterministic_parallel),
        "poker.dataloader_workers": args.dataloader_workers,
        "poker.torch_num_threads": args.torch_num_threads,
        "poker.require_cuda": int(args.require_cuda),
        "poker.rollout_batch_size": args.rollout_batch_size,
        "poker.use_torch_equity": int(args.use_torch_equity),
        "poker.torch_equity_device": args.torch_equity_device,
        "poker.use_amp": int(args.use_amp),
        "poker.use_torch_compile": int(args.use_torch_compile),
        "poker.use_reach_weighting": int(not args.disable_reach_weighting),
        "poker.reach_weight_mode": args.reach_weight_mode,
        "poker.reach_weight_clip": args.reach_weight_clip,
        "poker.reach_weight_auto_clip": int(not args.disable_reach_auto_clip),
        "poker.reach_weight_auto_quantile": args.reach_auto_clip_quantile,
        "poker.max_snapshot_cache": args.max_snapshot_cache,
        "poker.checkpoint_interval": args.checkpoint_interval,
        "poker.early_stop_patience": args.early_stop_patience,
        "poker.early_stop_min_iters": args.early_stop_min_iters,
        "poker.early_stop_entropy_floor": args.early_stop_entropy_floor,
        "poker.early_stop_entropy_patience": args.early_stop_entropy_patience,
        "poker.early_stop_regret_loss_ceiling": args.early_stop_regret_loss_ceiling,
        "poker.early_stop_policy_loss_ceiling": args.early_stop_policy_loss_ceiling,
        "poker.checkpoint_dir": args.checkpoint_dir,
        "poker.checkpoint_keep_last": args.checkpoint_keep_last,
    }


def log_mlflow_configuration(args, run_context, command):
    for name, value in build_mlflow_params(args, run_context).items():
        log_param(name, value)
    set_run_tags(build_mlflow_tags(args, run_context, command))


def main():
    args = parse_args()
    street_bet_multipliers = parse_street_multipliers(args.street_bet_multipliers)
    street_raise_multipliers = parse_street_multipliers(args.street_raise_multipliers)
    set_global_seed(args.seed, deterministic=args.deterministic)
    configure_torch_runtime(args.torch_num_threads, require_cuda=args.require_cuda)

    checkpoint_manager = CheckpointManager(
        root_dir=args.checkpoint_dir,
        experiment=args.experiment,
        run_name=args.run_name or None,
        resume_mode=args.resume_mode,
        seed=args.seed,
        keep_last=args.checkpoint_keep_last,
    )
    run_context = checkpoint_manager.prepare_run()
    args.run_name = run_context["run_name"]
    command = persist_run_command(args, args.command_file)

    env = PokerEnv(
        num_players=args.players,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        reward_unit="bb",
        street_bet_multipliers=street_bet_multipliers,
        street_raise_multipliers=street_raise_multipliers,
        seed=args.seed,
    )
    trainer = DeepCFRTrainer(
        env,
        mc_simulations=args.mc_simulations,
        lut_simulations=args.lut_simulations,
        lut_dir=args.lut_dir,
        seed=args.seed,
        rollout_samples_per_action=args.rollouts_per_action,
        feature_cache_size=args.feature_cache_size,
        batch_size=args.batch_size,
        regret_epochs=args.regret_epochs,
        policy_epochs=args.policy_epochs,
        grad_clip=args.gradient_clip,
        parallel_workers=args.parallel_workers,
        deterministic_parallel=(not args.disable_deterministic_parallel),
        snapshot_pool_size=args.snapshot_pool_size,
        max_snapshot_cache=args.max_snapshot_cache,
        population_run_limit=args.population_run_limit,
        population_checkpoint_name=args.population_checkpoint_name,
        population_mix_prob=args.population_mix_prob,
        checkpoint_manager=checkpoint_manager,
        checkpoint_interval=args.checkpoint_interval,
        policy_smoothing_alpha=args.policy_smoothing_alpha,
        entropy_regularization=args.entropy_regularization,
        dataloader_workers=args.dataloader_workers,
        rollout_batch_size=args.rollout_batch_size,
        use_torch_equity=args.use_torch_equity,
        torch_equity_device=args.torch_equity_device,
        use_amp=args.use_amp,
        use_torch_compile=args.use_torch_compile,
        use_reach_weighting=(not args.disable_reach_weighting),
        reach_weight_mode=args.reach_weight_mode,
        reach_weight_clip=args.reach_weight_clip,
        reach_weight_auto_clip=(not args.disable_reach_auto_clip),
        reach_weight_auto_quantile=args.reach_auto_clip_quantile,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    # --- INICIO DEL CAMBIO ---
    print(f"\n[Init] Preparando LUT preflop para {args.players} jugadores...")
    all_cards = Deck().cards
    trainer.equity_lut.warmup_preflop(all_cards=all_cards, max_players=args.players)
    print("[Init] Warmup completado con éxito.\n")
    # --- FIN DEL CAMBIO ---

    resume_payload = checkpoint_manager.load_resume_checkpoint(map_location=trainer.device)
    if resume_payload is not None:
        trainer.load_checkpoint(resume_payload)

    mlflow_run_id = None
    if run_context["is_resumed"]:
        mlflow_run_id = checkpoint_manager.get_mlflow_run_id()
    started_run_id = start_experiment_run(
        args.experiment,
        run_name=run_context["run_name"],
        run_id=mlflow_run_id,
    )
    checkpoint_manager.set_mlflow_run_id(started_run_id)

    run_status = "FAILED"
    try:
        log_mlflow_configuration(args, run_context, command)

        trainer.train(
            iterations=args.iterations,
            episodes=args.episodes,
            eval_hands=args.eval_hands,
            eval_random_hands=args.eval_random_hands,
            eval_snapshot_hands=args.eval_snapshot_hands,
            eval_population_hands=args.eval_population_hands,
            eval_heuristic_hands=args.eval_heuristic_hands,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_iters=args.early_stop_min_iters,
            early_stop_entropy_floor=args.early_stop_entropy_floor,
            early_stop_entropy_patience=args.early_stop_entropy_patience,
            early_stop_regret_loss_ceiling=args.early_stop_regret_loss_ceiling,
            early_stop_policy_loss_ceiling=args.early_stop_policy_loss_ceiling,
        )

        if checkpoint_manager.run_summary_path is not None and checkpoint_manager.run_summary_path.exists():
            log_artifact(str(checkpoint_manager.run_summary_path))

        set_run_tags({"poker.run_status": "finished"})
        run_status = "FINISHED"
        print("Training complete.")
    except Exception as exc:
        try:
            set_run_tags(
                {
                    "poker.run_status": "failed",
                    "poker.failure_type": type(exc).__name__,
                }
            )
        except Exception as tag_exc:
            print(f"[MLFLOW WARNING] failed to tag run failure metadata: {tag_exc}")
        raise
    finally:
        end_experiment(status=run_status)


if __name__ == "__main__":
    main()
