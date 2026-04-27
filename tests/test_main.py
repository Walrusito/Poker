from pathlib import Path
from types import SimpleNamespace

import pytest

import main


def _make_args(tmp_path):
    return SimpleNamespace(
        iterations=2,
        episodes=3,
        eval_hands=20,
        eval_random_hands=10,
        eval_snapshot_hands=8,
        eval_population_hands=6,
        eval_heuristic_hands=4,
        players=2,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        mc_simulations=16,
        lut_simulations=32,
        lut_dir=str(tmp_path / "lut"),
        street_bet_multipliers="",
        street_raise_multipliers="",
        rollouts_per_action=1,
        feature_cache_size=128,
        batch_size=16,
        regret_epochs=1,
        policy_epochs=1,
        gradient_clip=1.0,
        parallel_workers=2,
        dataloader_workers=1,
        torch_num_threads=2,
        require_cuda=False,
        rollout_batch_size=8,
        use_torch_equity=False,
        torch_equity_device="cpu",
        use_amp=False,
        use_torch_compile=False,
        snapshot_pool_size=2,
        max_snapshot_cache=4,
        population_run_limit=3,
        population_checkpoint_name="best.pt",
        population_mix_prob=0.0,
        policy_smoothing_alpha=0.05,
        entropy_regularization=0.03,
        checkpoint_interval=1,
        early_stop_patience=0,
        early_stop_min_iters=2,
        early_stop_entropy_floor=0.02,
        early_stop_entropy_patience=3,
        early_stop_regret_loss_ceiling=500.0,
        early_stop_policy_loss_ceiling=25.0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        run_name="",
        resume_mode="auto",
        seed=7,
        deterministic=False,
        checkpoint_keep_last=0,
        num_layers=2,
        hidden_dim=32,
        dropout=0.0,
        experiment="exp_test_main",
        command_file=str(tmp_path / "last_run_command.txt"),
    )


class _FakeDeck:
    def __init__(self):
        self.cards = [1, 2, 3, 4]


class _FakeEquityLut:
    def __init__(self):
        self.calls = []

    def warmup_preflop(self, all_cards, max_players):
        self.calls.append((tuple(all_cards), max_players))


class _FakeTrainer:
    should_fail = False
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.device = "cpu"
        self.equity_lut = _FakeEquityLut()
        self.loaded_payload = None
        self.train_kwargs = None
        _FakeTrainer.last_instance = self

    def load_checkpoint(self, payload):
        self.loaded_payload = payload

    def train(self, **kwargs):
        self.train_kwargs = dict(kwargs)
        if self.should_fail:
            raise RuntimeError("boom")


class _FakeCheckpointManager:
    last_instance = None

    def __init__(self, *args, **kwargs):
        checkpoint_dir = Path(kwargs["root_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_summary_path = checkpoint_dir / "run_summary.json"
        self.run_summary_path.write_text("{}", encoding="utf-8")
        self.saved_run_id = None
        _FakeCheckpointManager.last_instance = self

    def prepare_run(self):
        return {
            "run_name": "run_a",
            "run_dir": str(self.run_summary_path.parent / "run_a"),
            "is_resumed": False,
        }

    def load_resume_checkpoint(self, map_location="cpu"):
        return None

    def set_mlflow_run_id(self, run_id):
        self.saved_run_id = run_id

    def get_mlflow_run_id(self):
        return self.saved_run_id


class _FakeEnv:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


def test_main_marks_run_finished_and_logs_runtime_tags(monkeypatch, tmp_path):
    args = _make_args(tmp_path)
    calls = {"params": {}, "tags": [], "artifacts": [], "end_statuses": []}
    _FakeTrainer.should_fail = False

    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main, "set_global_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "configure_torch_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "CheckpointManager", _FakeCheckpointManager)
    monkeypatch.setattr(main, "PokerEnv", _FakeEnv)
    monkeypatch.setattr(main, "DeepCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(main, "Deck", _FakeDeck)
    monkeypatch.setattr(main, "persist_run_command", lambda args, path: "docker compose run --rm poker-ai")
    monkeypatch.setattr(main, "start_experiment_run", lambda *args, **kwargs: "mlflow-run-1")
    monkeypatch.setattr(main, "log_param", lambda name, value: calls["params"].setdefault(name, value))
    monkeypatch.setattr(main, "set_run_tags", lambda tags: calls["tags"].append(dict(tags)))
    monkeypatch.setattr(main, "log_artifact", lambda path: calls["artifacts"].append(path))
    monkeypatch.setattr(main, "end_experiment", lambda status="FINISHED": calls["end_statuses"].append(status))

    main.main()

    assert calls["end_statuses"] == ["FINISHED"]
    assert calls["params"]["players"] == 2
    assert calls["params"]["run_name"] == "run_a"
    assert any(batch.get("poker.command") == "docker compose run --rm poker-ai" for batch in calls["tags"])
    assert any(batch.get("poker.run_status") == "finished" for batch in calls["tags"])
    assert str(_FakeCheckpointManager.last_instance.run_summary_path) in calls["artifacts"]
    assert _FakeTrainer.last_instance.train_kwargs["iterations"] == 2


def test_main_marks_run_failed_when_training_raises(monkeypatch, tmp_path):
    args = _make_args(tmp_path)
    calls = {"tags": [], "end_statuses": []}
    _FakeTrainer.should_fail = True

    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main, "set_global_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "configure_torch_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "CheckpointManager", _FakeCheckpointManager)
    monkeypatch.setattr(main, "PokerEnv", _FakeEnv)
    monkeypatch.setattr(main, "DeepCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(main, "Deck", _FakeDeck)
    monkeypatch.setattr(main, "persist_run_command", lambda args, path: "docker compose run --rm poker-ai")
    monkeypatch.setattr(main, "start_experiment_run", lambda *args, **kwargs: "mlflow-run-2")
    monkeypatch.setattr(main, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "set_run_tags", lambda tags: calls["tags"].append(dict(tags)))
    monkeypatch.setattr(main, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "end_experiment", lambda status="FINISHED": calls["end_statuses"].append(status))

    with pytest.raises(RuntimeError, match="boom"):
        main.main()

    assert calls["end_statuses"] == ["FAILED"]
    assert any(batch.get("poker.run_status") == "failed" for batch in calls["tags"])
    assert any(batch.get("poker.failure_type") == "RuntimeError" for batch in calls["tags"])
