import torch

from env.poker_env import PokerEnv
from models.policy_net import PolicyNet
from train.train_deep_cfr import DeepCFRTrainer
from utils.checkpointing import CheckpointManager


def test_multiway_self_play_populates_buffers_and_stats(tmp_path):
    env = PokerEnv(num_players=3, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    trainer = DeepCFRTrainer(
        env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        rollout_samples_per_action=1,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        feature_cache_size=128,
        parallel_workers=2,
        seed=5,
    )

    stats = trainer.self_play(episodes=3)

    assert stats["visited_nodes"] > 0
    assert stats["avg_branching_factor"] >= 1.0
    assert len(trainer.advantage_buffer) > 0
    assert len(trainer.policy_buffer) > 0
    assert trainer.equity_lut is trainer.iss.card_abs.equity_provider


def test_multiway_evaluation_reports_all_seats(tmp_path):
    env = PokerEnv(num_players=4, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    checkpoint_manager = CheckpointManager(root_dir=tmp_path / "checkpoints", experiment="exp_test", run_name="run_a", resume_mode="never", seed=9)
    checkpoint_manager.prepare_run()
    trainer = DeepCFRTrainer(
        env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        rollout_samples_per_action=1,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        feature_cache_size=128,
        snapshot_pool_size=2,
        checkpoint_manager=checkpoint_manager,
        seed=9,
    )

    trainer.self_play(episodes=2)
    checkpoint_manager.save_checkpoint(trainer.state_dict(), iteration=1, metrics={"vs_random_bb_per_100": 1.5}, is_best=True)
    self_play_metrics = trainer.evaluate_self_play(num_hands=6)
    random_metrics = trainer.evaluate_against_random(num_hands=8)
    snapshot_metrics = trainer.evaluate_against_snapshot_pool(num_hands=8, before_iteration=2)

    assert len(self_play_metrics["seat_ev_bb_per_hand"]) == 4
    assert abs(sum(self_play_metrics["seat_ev_bb_per_hand"])) < 1e-5
    assert len(random_metrics["hero_seat_ev_bb_per_hand"]) == 4
    assert isinstance(random_metrics["hero_bb_per_100"], float)
    assert snapshot_metrics is not None
    assert snapshot_metrics["pool_size"] == 1


def test_evaluation_uses_isolated_env_state(tmp_path):
    env = PokerEnv(num_players=3, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    trainer = DeepCFRTrainer(
        env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        seed=15,
    )
    initial_hand_index = trainer.env.hand_index
    initial_button = trainer.env.position.button

    trainer.evaluate_self_play(num_hands=4)

    assert trainer.env.hand_index == initial_hand_index
    assert trainer.env.position.button == initial_button


def test_snapshot_loader_supports_custom_architecture_and_legacy_keys(tmp_path):
    env = PokerEnv(num_players=2, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    trainer = DeepCFRTrainer(
        env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        seed=21,
    )
    payload = trainer.state_dict()
    legacy_policy_state = {}
    for key, value in payload["policy_net_state"].items():
        legacy_key = key.replace("net.", "_backbone.", 1) if key.startswith("net.") else key
        legacy_policy_state[legacy_key] = value.clone()
    payload["policy_net_state"] = legacy_policy_state

    snapshot_path = tmp_path / "snapshot.pt"
    torch.save(payload, snapshot_path)

    loaded = trainer._load_snapshot_policy(snapshot_path)

    assert isinstance(loaded, PolicyNet)
    assert loaded.hidden_dim == 64
    assert loaded.num_layers == 3
    assert abs(loaded.dropout - 0.1) < 1e-9


def test_checkpoint_load_fails_on_feature_schema_mismatch(tmp_path):
    env = PokerEnv(num_players=2, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    trainer = DeepCFRTrainer(
        env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        seed=11,
    )
    payload = trainer.state_dict()
    payload["feature_schema"] = dict(payload["feature_schema"])
    payload["feature_schema"]["fingerprint"] = "broken-fingerprint"

    try:
        trainer.load_checkpoint(payload)
        assert False, "Expected feature schema mismatch to raise ValueError"
    except ValueError as exc:
        assert "feature schema mismatch" in str(exc).lower()


def test_checkpoint_load_fails_on_environment_mismatch(tmp_path):
    source_env = PokerEnv(num_players=4, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    source_trainer = DeepCFRTrainer(
        source_env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        seed=13,
    )
    payload = source_trainer.state_dict()

    target_env = PokerEnv(num_players=2, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    target_trainer = DeepCFRTrainer(
        target_env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        seed=17,
    )

    try:
        target_trainer.load_checkpoint(payload)
        assert False, "Expected environment configuration mismatch to raise ValueError"
    except ValueError as exc:
        assert "environment configuration mismatch" in str(exc).lower()


def test_checkpoint_load_fails_on_training_config_mismatch(tmp_path):
    source_env = PokerEnv(num_players=2, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    source_trainer = DeepCFRTrainer(
        source_env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        seed=19,
    )
    payload = source_trainer.state_dict()

    target_env = PokerEnv(num_players=2, starting_stack=2000, small_blind=10, big_blind=20, reward_unit="bb")
    target_trainer = DeepCFRTrainer(
        target_env,
        mc_simulations=8,
        lut_simulations=8,
        lut_dir=tmp_path,
        batch_size=32,
        regret_epochs=1,
        policy_epochs=1,
        seed=19,
    )

    try:
        target_trainer.load_checkpoint(payload)
        assert False, "Expected training configuration mismatch to raise ValueError"
    except ValueError as exc:
        assert "training configuration mismatch" in str(exc).lower()


def test_exploitability_proxy_is_finite():
    metrics = {
        "self_play_mean_abs_seat_ev": 0.1,
        "self_play_utility_sum_error": 0.0,
        "avg_abs_regret": 0.5,
    }
    value = DeepCFRTrainer._compute_exploitability_proxy(metrics)
    assert isinstance(value, float)
    assert value >= 0.0
