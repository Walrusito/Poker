from argparse import Namespace

from utils.command_persistence import persist_run_command


def test_command_is_persisted_to_file(tmp_path):
    args = Namespace(
        iterations=5,
        episodes=10,
        eval_hands=20,
        eval_random_hands=12,
        eval_snapshot_hands=8,
        eval_population_hands=6,
        eval_heuristic_hands=4,
        players=4,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        mc_simulations=200,
        lut_simulations=1200,
        lut_dir=str(tmp_path / "lut"),
        rollouts_per_action=2,
        feature_cache_size=256,
        batch_size=64,
        regret_epochs=3,
        policy_epochs=2,
        gradient_clip=1.0,
        parallel_workers=2,
        snapshot_pool_size=5,
        max_snapshot_cache=6,
        population_run_limit=3,
        population_checkpoint_name="best.pt",
        population_mix_prob=0.25,
        checkpoint_interval=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        run_name="",
        resume_mode="auto",
        seed=13,
        deterministic=True,
        experiment="exp_test",
        command_file=str(tmp_path / "last_run.txt"),
    )

    command = persist_run_command(args, args.command_file)
    stored = (tmp_path / "last_run.txt").read_text(encoding="utf-8").strip()

    assert command == stored
    assert "--players 4" in stored
    assert "--rollouts-per-action 2" in stored
    assert "--snapshot-pool-size 5" in stored
    assert "--seed 13" in stored
    assert "--deterministic" in stored
