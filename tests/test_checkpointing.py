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
