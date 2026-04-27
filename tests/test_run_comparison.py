from utils.checkpointing import CheckpointManager
from utils.run_comparison import collect_run_summaries, format_run_table, sort_run_summaries


def test_run_comparison_collects_and_sorts_runs(tmp_path):
    manager_a = CheckpointManager(root_dir=tmp_path, experiment="exp_test", run_name="run_a", resume_mode="never", seed=1)
    manager_a.prepare_run()
    manager_a.save_checkpoint({"policy_net_state": {}, "regret_net_state": {}}, iteration=1, metrics={"vs_random_bb_per_100": 10.0}, is_best=True)

    manager_b = CheckpointManager(root_dir=tmp_path, experiment="exp_test", run_name="run_b", resume_mode="never", seed=2)
    manager_b.prepare_run()
    manager_b.save_checkpoint({"policy_net_state": {}, "regret_net_state": {}}, iteration=1, metrics={"vs_random_bb_per_100": 20.0}, is_best=True)

    summaries = collect_run_summaries(tmp_path, "exp_test")
    sorted_summaries = sort_run_summaries(summaries, sort_key="best_vs_random_bb_per_100")
    table = format_run_table(sorted_summaries)

    assert len(summaries) == 2
    assert sorted_summaries[0]["run_name"] == "run_b"
    assert "run_b" in table
