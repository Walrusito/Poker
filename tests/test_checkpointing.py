from utils.checkpointing import CheckpointManager


def test_checkpoint_manager_creates_new_run_when_none_exists(tmp_path):
    manager = CheckpointManager(root_dir=tmp_path, experiment="exp_test", resume_mode="auto", seed=17)
    context = manager.prepare_run()

    assert context["is_resumed"] is False
    assert manager.run_dir.exists()
    assert manager.run_summary_path.exists()


def test_checkpoint_manager_resumes_latest_run_when_checkpoint_exists(tmp_path):
    manager = CheckpointManager(root_dir=tmp_path, experiment="exp_test", resume_mode="auto", seed=17)
    manager.prepare_run()
    manager.save_checkpoint({"policy_net_state": {}, "regret_net_state": {}}, iteration=1, metrics={"vs_random_bb_per_100": 12.5}, is_best=True)

    followup = CheckpointManager(root_dir=tmp_path, experiment="exp_test", resume_mode="auto", seed=17)
    context = followup.prepare_run()

    assert context["is_resumed"] is True
    assert followup.resume_checkpoint_path == followup.latest_checkpoint_path
    assert followup.best_checkpoint_path.exists()


def test_checkpoint_manager_tracks_best_robust_checkpoint(tmp_path):
    manager = CheckpointManager(root_dir=tmp_path, experiment="exp_test", resume_mode="auto", seed=17)
    manager.prepare_run()
    payload = {"policy_net_state": {}, "regret_net_state": {}}
    manager.save_checkpoint(payload, iteration=1, metrics={"vs_random_bb_per_100": 1.0, "robust_score": 2.5}, is_best=False, is_best_robust=True)

    summary = manager._read_summary()
    assert manager.best_robust_checkpoint_path.exists()
    assert summary["best_robust_score"] == 2.5
    assert summary["best_robust_iteration"] == 1


def test_checkpoint_manager_writes_lightweight_policy_snapshots(tmp_path):
    manager = CheckpointManager(root_dir=tmp_path, experiment="exp_test", run_name="run_a", resume_mode="never", seed=17)
    manager.prepare_run()
    payload = {
        "trainer_version": 6,
        "seed": 17,
        "actions": ["fold", "call"],
        "env_config": {"num_players": 2},
        "feature_schema": {"fingerprint": "abc"},
        "config": {"input_dim": 31, "output_dim": 2, "hidden_dim": 32, "num_layers": 2, "dropout": 0.0},
        "policy_net_state": {},
        "regret_net_state": {},
    }

    manager.save_checkpoint(payload, iteration=1, metrics={"vs_random_bb_per_100": 3.5}, is_best=True, is_best_robust=True)

    snapshot_paths = manager.list_snapshot_paths(limit=1)

    assert manager.latest_policy_checkpoint_path.exists()
    assert manager.best_policy_checkpoint_path.exists()
    assert manager.best_robust_policy_checkpoint_path.exists()
    assert snapshot_paths
    assert snapshot_paths[0].name == "policy_iter_0001.pt"
