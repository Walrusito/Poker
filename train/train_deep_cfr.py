import copy
import math
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from data.buffers import ReservoirBuffer
from data.dataset import AdvantageDataset, PolicyDataset
from models.policy_net import PolicyNet
from models.regret_net import RegretNet
from env.vectorized_poker_env import VectorizedPokerEnv
from utils.action_abstraction import DEFAULT_ACTIONS
from utils.information_set import InformationSetBuilder
from utils.logging import log_metric, log_metrics_batch
from utils.run_comparison import collect_population_checkpoints


class DeepCFRTrainer:
    IMPORTANT_METRIC_KEYS = (
        "vs_random_bb_per_100",
        "vs_random_bb_per_100_stderr",
        "vs_heuristic_bb_per_100",
        "vs_snapshot_bb_per_100",
        "vs_population_bb_per_100",
        "robust_score",
        "exploitability_proxy",
        "regret_loss",
        "policy_loss",
        "avg_policy_entropy",
        "avg_abs_regret",
        "br_gap_proxy",
        "reach_weight_clip_optimal",
        "self_play_showdown_rate",
        "lut_hit_rate",
        "equity_avg_ms",
    )
    CHECKPOINT_HISTORY_KEYS = (
        "robust_score",
        "exploitability_proxy",
        "vs_random_bb_per_100",
        "vs_snapshot_bb_per_100",
        "vs_population_bb_per_100",
        "vs_heuristic_bb_per_100",
        "vs_heuristic_pool_bb_per_100",
        "avg_policy_entropy",
        "regret_loss",
        "policy_loss",
    )
    RESUME_VALIDATION_CONFIG_KEYS = (
        "rollout_samples_per_action",
        "batch_size",
        "regret_epochs",
        "policy_epochs",
        "grad_clip",
        "policy_smoothing_alpha",
        "entropy_regularization",
        "population_mix_prob",
        "mc_simulations",
        "lut_simulations",
        "use_reach_weighting",
        "reach_weight_mode",
        "reach_weight_clip",
        "reach_weight_auto_clip",
        "reach_weight_auto_quantile",
    )

    def __init__(
        self,
        env,
        mc_simulations: int = 200,
        lut_simulations: int = 1500,
        lut_dir: str = "data/lut",
        seed=None,
        rollout_samples_per_action: int = 1,
        feature_cache_size: int = 50000,
        batch_size: int = 128,
        regret_epochs: int = 4,   # Train regret net more: it gets better signal
        policy_epochs: int = 1,   # Train policy net LESS: prevents entropy collapse
        grad_clip: float = 1.0,
        parallel_workers: int = 1,
        snapshot_pool_size: int = 4,
        max_snapshot_cache: int = 8,
        population_run_limit: int = 6,
        population_checkpoint_name: str = "best.pt",
        population_mix_prob: float = 0.0,
        checkpoint_manager=None,
        checkpoint_interval: int = 1,
        policy_smoothing_alpha: float = 0.05,
        entropy_regularization: float = 0.01,
        dataloader_workers: int = 0,
        rollout_batch_size: int = 32,
        use_torch_equity: bool = False,
        torch_equity_device: str = "cuda",
        use_amp: bool = True,
        use_torch_compile: bool = False,
        use_reach_weighting: bool = True,
        reach_weight_mode: str = "linear",
        reach_weight_clip: float = 100.0,
        reach_weight_auto_clip: bool = True,
        reach_weight_auto_quantile: float = 0.95,
        deterministic_parallel: bool = True,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        self.env = env
        self.seed = seed
        self.iss = InformationSetBuilder(
            mc_simulations=mc_simulations,
            lut_simulations=lut_simulations,
            lut_dir=lut_dir,
            seed=seed,
            cache_size=feature_cache_size,
            use_torch_backend=use_torch_equity,
            torch_device=torch_equity_device,
        )
        # Shared equity provider used by the feature encoder and optional LUT warmup.
        self.equity_lut = self.iss.card_abs.equity_provider
        self.actions = tuple(getattr(env, "ACTIONS", DEFAULT_ACTIONS))
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = self.iss.feature_dim
        self.num_actions = len(self.actions)
        self.rollout_samples_per_action = max(1, int(rollout_samples_per_action))
        self.batch_size = max(8, int(batch_size))
        self.regret_epochs = max(1, int(regret_epochs))
        self.policy_epochs = max(1, int(policy_epochs))
        self.grad_clip = max(0.0, float(grad_clip))
        self.parallel_workers = max(1, int(parallel_workers))
        self.snapshot_pool_size = max(0, int(snapshot_pool_size))
        self.max_snapshot_cache = max(1, int(max_snapshot_cache))
        self.population_run_limit = max(0, int(population_run_limit))
        self.population_checkpoint_name = population_checkpoint_name
        self.population_mix_prob = max(0.0, min(1.0, float(population_mix_prob)))
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.policy_smoothing_alpha = max(0.0, min(0.5, float(policy_smoothing_alpha)))
        self.entropy_regularization = max(0.0, float(entropy_regularization))
        self.dataloader_workers = max(0, int(dataloader_workers))
        self.rollout_batch_size = max(1, int(rollout_batch_size))
        self.use_torch_equity = bool(use_torch_equity)
        self.torch_equity_device = torch_equity_device
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self.use_reach_weighting = bool(use_reach_weighting)
        normalized_mode = str(reach_weight_mode).strip().lower()
        self.reach_weight_mode = normalized_mode if normalized_mode in {"linear", "sqrt"} else "linear"
        self.reach_weight_clip = max(1.0, float(reach_weight_clip))
        self.reach_weight_auto_clip = bool(reach_weight_auto_clip)
        self.reach_weight_auto_quantile = min(0.999, max(0.50, float(reach_weight_auto_quantile)))
        self.effective_reach_weight_clip = self.reach_weight_clip
        self._reach_weight_observations = []
        self.deterministic_parallel = bool(deterministic_parallel)
        self.num_layers = max(1, int(num_layers))
        self.hidden_dim = max(1, int(hidden_dim))
        self.dropout = max(0.0, float(dropout))
        # torch.compile + multi-threaded inference can trigger Dynamo/Fx tracing conflicts.
        # We disable compile in that configuration for stability.
        requested_compile = bool(use_torch_compile)
        self.use_torch_compile = requested_compile and (self.parallel_workers <= 1)
        self.rng = np.random.default_rng(seed)
        self._rng_lock = threading.Lock()
        self._snapshot_cache_lock = threading.RLock()

        self.regret_net = RegretNet(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim,
            output_dim=self.num_actions, num_layers=self.num_layers, dropout=self.dropout,
        ).to(self.device)
        self.policy_net = PolicyNet(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim,
            output_dim=self.num_actions, num_layers=self.num_layers, dropout=self.dropout,
        ).to(self.device)
        if requested_compile and not self.use_torch_compile:
            print("[Runtime] torch.compile disabled (parallel_workers>1).")

        if self.use_torch_compile and hasattr(torch, "compile"):
            self.regret_net = torch.compile(self.regret_net)  # type: ignore[attr-defined]
            self.policy_net = torch.compile(self.policy_net)  # type: ignore[attr-defined]

        self.advantage_buffer = ReservoirBuffer(max_size=100000)
        self.policy_buffer = ReservoirBuffer(max_size=100000)

        self.regret_opt = optim.Adam(self.regret_net.parameters(), lr=5e-3)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.regret_loss_fn = nn.SmoothL1Loss()
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self._scaler = GradScaler(enabled=self.use_amp)

        self.completed_iterations = 0
        self.best_vs_random_bb_per_100 = float("-inf")
        self.metrics_history = []
        self.snapshot_policy_cache = OrderedDict()
        self.population_checkpoint_paths = []
        self._reset_traversal_stats()

    @staticmethod
    def _require_finite_tensor(tensor, name: str):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values detected in tensor: {name}")

    @staticmethod
    def _require_finite_scalar(value: float, name: str):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite scalar detected: {name}={value}")

    @staticmethod
    def _env_config_snapshot(env):
        bet_sizing = getattr(env, "bet_sizing", None)
        return {
            "num_players": int(env.num_players),
            "starting_stack": float(env.starting_stack),
            "small_blind": float(env.small_blind),
            "big_blind": float(env.big_blind),
            "reward_unit": env.reward_unit,
            "actions": list(getattr(env, "ACTIONS", ())),
            "street_bet_multipliers": copy.deepcopy(getattr(bet_sizing, "street_bet_multipliers", None)),
            "street_raise_multipliers": copy.deepcopy(getattr(bet_sizing, "street_raise_multipliers", None)),
        }

    @classmethod
    def _checkpoint_env_config(cls, payload):
        checkpoint_config = payload.get("env_config")
        if checkpoint_config is not None:
            return checkpoint_config

        checkpoint_env = payload.get("env_state")
        if checkpoint_env is None:
            return None
        return cls._env_config_snapshot(checkpoint_env)

    @staticmethod
    def _validate_resume_env(current_config, checkpoint_config):
        if checkpoint_config is None:
            return

        mismatches = []
        for key in (
            "num_players",
            "starting_stack",
            "small_blind",
            "big_blind",
            "reward_unit",
            "actions",
            "street_bet_multipliers",
            "street_raise_multipliers",
        ):
            if current_config.get(key) != checkpoint_config.get(key):
                mismatches.append(
                    f"{key}: current={current_config.get(key)!r} checkpoint={checkpoint_config.get(key)!r}"
                )

        if mismatches:
            raise ValueError(
                "Checkpoint environment configuration mismatch. "
                + "; ".join(mismatches)
            )

    def _policy_arch_config(self):
        return {
            "input_dim": int(self.input_dim),
            "output_dim": int(self.num_actions),
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
        }

    def _resume_validation_config(self):
        return {
            "rollout_samples_per_action": int(self.rollout_samples_per_action),
            "batch_size": int(self.batch_size),
            "regret_epochs": int(self.regret_epochs),
            "policy_epochs": int(self.policy_epochs),
            "grad_clip": float(self.grad_clip),
            "policy_smoothing_alpha": float(self.policy_smoothing_alpha),
            "entropy_regularization": float(self.entropy_regularization),
            "population_mix_prob": float(self.population_mix_prob),
            "mc_simulations": int(self.equity_lut.fallback_equity.simulations),
            "lut_simulations": int(self.equity_lut.lut_equity.simulations),
            "use_reach_weighting": int(self.use_reach_weighting),
            "reach_weight_mode": self.reach_weight_mode,
            "reach_weight_clip": float(self.reach_weight_clip),
            "reach_weight_auto_clip": int(self.reach_weight_auto_clip),
            "reach_weight_auto_quantile": float(self.reach_weight_auto_quantile),
        }

    @classmethod
    def _validate_resume_config(cls, current_config, checkpoint_config):
        if checkpoint_config is None:
            return

        mismatches = []
        for key in cls.RESUME_VALIDATION_CONFIG_KEYS:
            checkpoint_value = checkpoint_config.get(key)
            if checkpoint_value is None:
                continue
            current_value = current_config.get(key)
            if current_value != checkpoint_value:
                mismatches.append(
                    f"{key}: current={current_value!r} checkpoint={checkpoint_value!r}"
                )

        if mismatches:
            raise ValueError(
                "Checkpoint training configuration mismatch. "
                + "; ".join(mismatches)
            )

    @classmethod
    def _compact_metrics_entry(cls, iteration_number, metrics):
        entry = {"iteration": int(iteration_number)}
        for key in cls.CHECKPOINT_HISTORY_KEYS:
            value = metrics.get(key)
            if value is not None:
                entry[key] = value
        return entry

    def _make_isolated_env(self, seed=None):
        bet_sizing = getattr(self.env, "bet_sizing", None)
        env_cls = self.env.__class__
        return env_cls(
            num_players=int(self.env.num_players),
            starting_stack=float(self.env.starting_stack),
            small_blind=float(self.env.small_blind),
            big_blind=float(self.env.big_blind),
            reward_unit=self.env.reward_unit,
            street_bet_multipliers=copy.deepcopy(getattr(bet_sizing, "street_bet_multipliers", None)),
            street_raise_multipliers=copy.deepcopy(getattr(bet_sizing, "street_raise_multipliers", None)),
            seed=self.seed if seed is None else int(seed),
        )

    def state_dict(self):
        return {
            "trainer_version": 6,
            "checkpoint_type": "full",
            "seed": self.seed,
            "completed_iterations": self.completed_iterations,
            "best_vs_random_bb_per_100": self.best_vs_random_bb_per_100,
            "metrics_history": self.metrics_history,
            "rng_state": self.rng.bit_generator.state,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "actions": list(self.actions),
            "env_config": self._env_config_snapshot(self.env),
            "regret_net_state": self.regret_net.state_dict(),
            "policy_net_state": self.policy_net.state_dict(),
            "regret_opt_state": self.regret_opt.state_dict(),
            "policy_opt_state": self.policy_opt.state_dict(),
            "advantage_buffer": {
                "data": self.advantage_buffer.data,
                "n_seen": self.advantage_buffer.n_seen,
                "max_size": self.advantage_buffer.max_size,
            },
            "policy_buffer": {
                "data": self.policy_buffer.data,
                "n_seen": self.policy_buffer.n_seen,
                "max_size": self.policy_buffer.max_size,
            },
            "config": {
                "rollout_samples_per_action": self.rollout_samples_per_action,
                "batch_size": self.batch_size,
                "regret_epochs": self.regret_epochs,
                "policy_epochs": self.policy_epochs,
                "grad_clip": self.grad_clip,
                "parallel_workers": self.parallel_workers,
                "snapshot_pool_size": self.snapshot_pool_size,
                "population_run_limit": self.population_run_limit,
                "population_checkpoint_name": self.population_checkpoint_name,
                "population_mix_prob": self.population_mix_prob,
                "checkpoint_interval": self.checkpoint_interval,
                "policy_smoothing_alpha": self.policy_smoothing_alpha,
                "entropy_regularization": self.entropy_regularization,
                "dataloader_workers": self.dataloader_workers,
                "rollout_batch_size": self.rollout_batch_size,
                "use_torch_equity": int(self.use_torch_equity),
                "torch_equity_device": self.torch_equity_device,
                "use_amp": int(self.use_amp),
                "use_torch_compile": int(self.use_torch_compile),
                "feature_cache_size": self.iss.cache_size,
                "mc_simulations": int(self.equity_lut.fallback_equity.simulations),
                "lut_simulations": int(self.equity_lut.lut_equity.simulations),
                "use_reach_weighting": int(self.use_reach_weighting),
                "reach_weight_mode": self.reach_weight_mode,
                "reach_weight_clip": float(self.reach_weight_clip),
                "reach_weight_auto_clip": int(self.reach_weight_auto_clip),
                "reach_weight_auto_quantile": float(self.reach_weight_auto_quantile),
                "deterministic_parallel": int(self.deterministic_parallel),
                "input_dim": self.input_dim,
                "output_dim": self.num_actions,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            "feature_schema": dict(self.iss.feature_schema),
        }

    def load_checkpoint(self, payload):
        checkpoint_actions = tuple(payload.get("actions", self.actions))
        if checkpoint_actions != self.actions:
            raise ValueError(f"Checkpoint actions {checkpoint_actions} do not match current actions {self.actions}")
        checkpoint_schema = payload.get("feature_schema")
        checkpoint_env_config = self._checkpoint_env_config(payload)
        current_schema = self.iss.feature_schema
        if checkpoint_schema is not None:
            if checkpoint_schema.get("fingerprint") != current_schema.get("fingerprint"):
                raise ValueError(
                    "Checkpoint feature schema mismatch. "
                    f"checkpoint={checkpoint_schema.get('fingerprint')} current={current_schema.get('fingerprint')}. "
                    "Start a new run/checkpoint set for this feature-space version."
                )
        self._validate_resume_env(self._env_config_snapshot(self.env), checkpoint_env_config)
        checkpoint_config = dict(payload.get("config") or {})
        self._validate_resume_config(self._resume_validation_config(), checkpoint_config)
        for key, current_value in self._policy_arch_config().items():
            checkpoint_value = checkpoint_config.get(key)
            if checkpoint_value is None:
                continue
            if checkpoint_value != current_value:
                raise ValueError(
                    "Checkpoint policy architecture mismatch. "
                    f"{key}: current={current_value!r} checkpoint={checkpoint_value!r}"
                )

        self.regret_net.load_state_dict(payload["regret_net_state"])
        self.policy_net.load_state_dict(payload["policy_net_state"])
        self.regret_opt.load_state_dict(payload["regret_opt_state"])
        self.policy_opt.load_state_dict(payload["policy_opt_state"])

        advantage_state = payload.get("advantage_buffer", {})
        self.advantage_buffer.max_size = int(advantage_state.get("max_size", self.advantage_buffer.max_size))
        self.advantage_buffer.data = advantage_state.get("data", [])
        self.advantage_buffer.n_seen = int(advantage_state.get("n_seen", len(self.advantage_buffer.data)))

        policy_state = payload.get("policy_buffer", {})
        self.policy_buffer.max_size = int(policy_state.get("max_size", self.policy_buffer.max_size))
        self.policy_buffer.data = policy_state.get("data", [])
        self.policy_buffer.n_seen = int(policy_state.get("n_seen", len(self.policy_buffer.data)))

        self.completed_iterations = int(payload.get("completed_iterations", 0))
        self.best_vs_random_bb_per_100 = float(payload.get("best_vs_random_bb_per_100", float("-inf")))
        self.metrics_history = list(payload.get("metrics_history", []))

        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = rng_state

        python_random_state = payload.get("python_random_state")
        if python_random_state is not None:
            random.setstate(python_random_state)

        numpy_random_state = payload.get("numpy_random_state")
        if numpy_random_state is not None:
            np.random.set_state(numpy_random_state)

        torch_rng_state = payload.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)

        torch_cuda_rng_state = payload.get("torch_cuda_rng_state")
        if torch_cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_rng_state)

        if payload.get("env_state") is not None:
            self.env = payload["env_state"]

    def get_state_tensor(self, state, player):
        return torch.tensor(
            self.iss.encode_vector(state, player),
            dtype=torch.float32,
            device=self.device,
        )

    def _next_seed(self):
        with self._rng_lock:
            return int(self.rng.integers(0, 2**32 - 1))

    def _spawn_rng(self, seed=None):
        base_seed = self._next_seed() if seed is None else int(seed)
        return np.random.default_rng(base_seed)

    @staticmethod
    def _child_seed(master_rng):
        return int(master_rng.integers(0, 2**32 - 1))

    def _refresh_population_pool(self):
        if self.checkpoint_manager is None or self.population_run_limit <= 0:
            self.population_checkpoint_paths = []
            return self.population_checkpoint_paths

        self.population_checkpoint_paths = collect_population_checkpoints(
            root_dir=str(self.checkpoint_manager.root_dir),
            experiment=self.checkpoint_manager.experiment,
            checkpoint_name=self.population_checkpoint_name,
            exclude_run_dir=str(self.checkpoint_manager.run_dir),
            limit=self.population_run_limit,
        )
        return self.population_checkpoint_paths

    def _combined_population_paths(self, before_iteration=None):
        candidate_paths = []

        if self.checkpoint_manager is not None and self.snapshot_pool_size > 0:
            candidate_paths.extend(
                str(path)
                for path in self.checkpoint_manager.list_snapshot_paths(
                    before_iteration=before_iteration,
                    limit=self.snapshot_pool_size,
                )
            )

        if self.population_checkpoint_paths:
            candidate_paths.extend(self.population_checkpoint_paths)

        unique_paths = []
        seen = set()
        for path in candidate_paths:
            if path in seen:
                continue
            unique_paths.append(path)
            seen.add(path)
        return unique_paths

    def _empty_stats(self):
        return {
            "visited_nodes": 0,
            "legal_action_total": 0.0,
            "entropy_total": 0.0,
            "regret_abs_total": 0.0,
            "regret_count": 0,
            "reach_cf_weight_total": 0.0,
            "reach_policy_weight_total": 0.0,
            "reach_weight_count": 0,
        }

    def _reset_traversal_stats(self):
        self._traversal_stats = self._empty_stats()

    def _merge_traversal_stats(self, local_stats):
        for key in self._traversal_stats:
            self._traversal_stats[key] += local_stats.get(key, 0.0)

    def _summarize_traversal_stats(self):
        visited_nodes = max(1, self._traversal_stats["visited_nodes"])
        regret_count = max(1, self._traversal_stats["regret_count"])
        reach_weight_count = max(1, self._traversal_stats["reach_weight_count"])
        return {
            "visited_nodes": self._traversal_stats["visited_nodes"],
            "avg_branching_factor": self._traversal_stats["legal_action_total"] / visited_nodes,
            "avg_policy_entropy": self._traversal_stats["entropy_total"] / visited_nodes,
            "avg_abs_regret": self._traversal_stats["regret_abs_total"] / regret_count,
            "reach_cf_weight_mean": self._traversal_stats["reach_cf_weight_total"] / reach_weight_count,
            "reach_policy_weight_mean": self._traversal_stats["reach_policy_weight_total"] / reach_weight_count,
        }

    def _legal_mask(self, legal_actions):
        return np.array([1.0 if action in legal_actions else 0.0 for action in self.actions], dtype=np.float32)

    def _normalize_strategy(self, raw_values, legal_actions, positive_only: bool = False):
        values = np.asarray(raw_values, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Non-finite strategy/regret values encountered")
        if positive_only:
            values = np.maximum(values, 0.0)

        mask = self._legal_mask(legal_actions)
        masked_values = values * mask
        total = float(np.sum(masked_values))

        if total > 0.0:
            return masked_values / total

        legal_count = float(np.sum(mask))
        if legal_count <= 0.0:
            raise ValueError(f"No legal actions available: {legal_actions}")

        return mask / legal_count

    def _regret_matching(self, legal_actions, predicted_regrets):
        return self._normalize_strategy(predicted_regrets, legal_actions, positive_only=True)

    def _strategy_entropy(self, strategy, legal_actions):
        legal_indices = [self.action_to_index[action] for action in legal_actions]
        legal_probs = strategy[legal_indices]
        legal_probs = legal_probs[legal_probs > 0.0]
        if len(legal_probs) == 0:
            return 0.0
        return float(-np.sum(legal_probs * np.log(legal_probs)))

    def _sample_action(self, strategy, legal_actions, rng):
        legal_indices = [self.action_to_index[action] for action in legal_actions]
        legal_probs = strategy[legal_indices]
        total = float(np.sum(legal_probs))

        if total <= 0.0:
            legal_probs = np.full(len(legal_indices), 1.0 / len(legal_indices), dtype=np.float32)
        else:
            legal_probs = legal_probs / total

        choice_idx = int(rng.choice(len(legal_indices), p=legal_probs))
        action_idx = legal_indices[choice_idx]
        return action_idx, self.actions[action_idx]

    @staticmethod
    def _sorted_bet_actions(legal_actions):
        def _rank(action_name):
            if not action_name.startswith("bet_"):
                return 10_000
            try:
                return int(action_name.split("_", 1)[1])
            except ValueError:
                return 10_000

        return sorted([action for action in legal_actions if action.startswith("bet_")], key=_rank)

    def _terminal_utilities(self, env, info=None):
        if info is not None and "terminal_utilities" in info:
            raw = np.asarray(info["terminal_utilities"], dtype=np.float32)
        else:
            raw = np.asarray(env.get_terminal_utilities(), dtype=np.float32)
        # Preserve the environment's raw zero-sum utilities for both training
        # and evaluation. Per-seat clipping here would distort EVs and can break
        # chip conservation in multiway pots.
        return raw

    def _policy_strategy_from_net(self, policy_net, state, player, legal_actions):
        x = self.get_state_tensor(state, player)
        with torch.inference_mode():
            predicted = policy_net(x.unsqueeze(0)).squeeze(0).cpu().numpy()
        strategy = self._normalize_strategy(predicted, legal_actions)
        legal_mask = self._legal_mask(legal_actions)
        legal_count = float(np.sum(legal_mask))
        if legal_count > 0 and self.policy_smoothing_alpha > 0.0:
            uniform = legal_mask / legal_count
            strategy = (1.0 - self.policy_smoothing_alpha) * strategy + self.policy_smoothing_alpha * uniform
            strategy = self._normalize_strategy(strategy, legal_actions)
        return strategy

    def _policy_rollout(self, env, rng, policy_net=None, population_paths=None, population_mix_prob=0.0):
        active_policy = self.policy_net if policy_net is None else policy_net

        while True:
            state = env._get_state()
            if state.get("done", False) or state["street"] == "showdown":
                return self._terminal_utilities(env)

            player = state["current_player"]
            legal_actions = env.get_legal_actions()
            rollout_policy = active_policy
            if population_paths and population_mix_prob > 0.0 and float(rng.random()) < population_mix_prob:
                sampled_path = population_paths[int(rng.integers(len(population_paths)))]
                rollout_policy = self._load_snapshot_policy(sampled_path)

            strategy = self._policy_strategy_from_net(rollout_policy, state, player, legal_actions)
            _, action = self._sample_action(strategy, legal_actions, rng)

            _, _, done, info = env.step(action)
            if done:
                return self._terminal_utilities(env, info)

    def _policy_rollout_batch(self, envs, rng, policy_net=None):
        active_policy = self.policy_net if policy_net is None else policy_net
        vec_env = VectorizedPokerEnv(envs)
        terminal_utilities = [None for _ in vec_env.envs]

        while True:
            active_indices = vec_env.active_indices()
            if not active_indices:
                break

            grouped = {}
            for idx in active_indices:
                state = vec_env.envs[idx]._get_state()
                if state.get("done", False) or state["street"] == "showdown":
                    terminal_utilities[idx] = self._terminal_utilities(vec_env.envs[idx])
                    continue
                player = state["current_player"]
                legal_actions = vec_env.envs[idx].get_legal_actions()
                grouped[idx] = (state, player, legal_actions)

            if not grouped:
                break

            state_tensors = []
            grouped_indices = []
            for idx, (state, player, _) in grouped.items():
                state_tensors.append(self.get_state_tensor(state, player))
                grouped_indices.append(idx)
            batch_x = torch.stack(state_tensors, dim=0)
            with torch.inference_mode():
                predicted = active_policy(batch_x).cpu().numpy()

            for row_idx, idx in enumerate(grouped_indices):
                _, _, legal_actions = grouped[idx]
                strategy = self._normalize_strategy(predicted[row_idx], legal_actions)
                _, action = self._sample_action(strategy, legal_actions, rng)
                _, _, done, info = vec_env.envs[idx].step(action)
                if done:
                    terminal_utilities[idx] = self._terminal_utilities(vec_env.envs[idx], info)

        return [utility for utility in terminal_utilities]

    def _estimate_action_utility(self, env, action, seed):
        total_utilities = np.zeros(env.num_players, dtype=np.float32)
        local_rng = np.random.default_rng(seed)
        population_paths = self._combined_population_paths()
        pending_envs = []

        for _ in range(self.rollout_samples_per_action):
            next_env = env.clone()
            _, _, done, info = next_env.step(action)
            if done:
                total_utilities += self._terminal_utilities(next_env, info)
            else:
                pending_envs.append(next_env)

        if pending_envs:
            for chunk_start in range(0, len(pending_envs), self.rollout_batch_size):
                chunk = pending_envs[chunk_start:chunk_start + self.rollout_batch_size]
                if population_paths and self.population_mix_prob > 0.0:
                    # Fallback to single rollout when mixing with snapshot policies.
                    for single_env in chunk:
                        total_utilities += self._policy_rollout(
                            single_env,
                            local_rng,
                            population_paths=population_paths,
                            population_mix_prob=self.population_mix_prob,
                        )
                else:
                    utilities = self._policy_rollout_batch(chunk, local_rng)
                    for utility in utilities:
                        total_utilities += utility

        return total_utilities / float(self.rollout_samples_per_action)

    def _collect_episode(self, episode_env, seed):
        local_rng = np.random.default_rng(seed)
        advantage_records = []
        policy_records = []
        local_stats = self._empty_stats()
        initial_reach = np.ones(episode_env.num_players, dtype=np.float64)
        self._traverse(
            episode_env,
            advantage_records,
            policy_records,
            local_stats,
            local_rng,
            initial_reach,
        )
        return advantage_records, policy_records, local_stats

    def self_play(self, episodes: int = 50):
        self._reset_traversal_stats()
        episode_envs = []
        seeds = []

        for _ in range(episodes):
            self.env.reset()
            episode_envs.append(self.env.clone())
            seeds.append(self._next_seed())

        progress_every = max(1, episodes // 5)

        if self.parallel_workers > 1 and len(episode_envs) > 1:
            max_workers = min(self.parallel_workers, len(episode_envs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if self.deterministic_parallel:
                    iterator = executor.map(self._collect_episode, episode_envs, seeds)
                    results = []
                    for idx, result in enumerate(iterator, start=1):
                        results.append(result)
                        if idx % progress_every == 0 or idx == episodes:
                            print(f"[DeepCFR] self-play progress: {idx}/{episodes} episodes")
                else:
                    results = []
                    futures = []
                    try:
                        for env_snapshot, seed in zip(episode_envs, seeds):
                            futures.append(executor.submit(self._collect_episode, env_snapshot, seed))
                        completed = 0
                        for fut in as_completed(futures):
                            results.append(fut.result())
                            completed += 1
                            if completed % progress_every == 0 or completed == episodes:
                                print(f"[DeepCFR] self-play progress: {completed}/{episodes} episodes")
                    except KeyboardInterrupt:
                        for fut in futures:
                            fut.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
        else:
            results = []
            for idx, (env_snapshot, seed) in enumerate(zip(episode_envs, seeds), start=1):
                results.append(self._collect_episode(env_snapshot, seed))
                if idx % progress_every == 0 or idx == episodes:
                    print(f"[DeepCFR] self-play progress: {idx}/{episodes} episodes")

        for advantage_records, policy_records, local_stats in results:
            for record in advantage_records:
                self.advantage_buffer.add(record)
            for record in policy_records:
                self.policy_buffer.add(record)
            self._merge_traversal_stats(local_stats)

        return self._summarize_traversal_stats()

    def _evaluate_other_actions(self, env, legal_actions, chosen_action):
        alternative_actions = [action for action in legal_actions if action != chosen_action]
        action_values = {}

        if not alternative_actions:
            return action_values

        if self.parallel_workers > 1 and len(alternative_actions) > 1:
            seeds = [self._next_seed() for _ in alternative_actions]
            max_workers = min(self.parallel_workers, len(alternative_actions))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._estimate_action_utility, [env] * len(alternative_actions), alternative_actions, seeds))
            for action, utility in zip(alternative_actions, results):
                action_values[action] = utility
        else:
            for action in alternative_actions:
                action_values[action] = self._estimate_action_utility(env, action, self._next_seed())

        return action_values

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor):
        safe_weights = torch.clamp(weights, min=1e-8)
        return torch.sum(values * safe_weights) / torch.sum(safe_weights)

    def _observe_reach_weight(self, weight: float):
        if not math.isfinite(weight):
            return
        max_samples = 100000
        if len(self._reach_weight_observations) < max_samples:
            self._reach_weight_observations.append(float(weight))
        # Keep deterministic and lock-free behavior in parallel mode.

    def _refresh_effective_reach_clip(self):
        if not self.use_reach_weighting:
            self.effective_reach_weight_clip = 1.0
            return
        if not self.reach_weight_auto_clip or not self._reach_weight_observations:
            self.effective_reach_weight_clip = self.reach_weight_clip
            return
        observed = np.asarray(self._reach_weight_observations, dtype=np.float64)
        quantile_clip = float(np.quantile(observed, self.reach_weight_auto_quantile))
        bounded_clip = max(1e-6, min(1.0, quantile_clip))
        self.effective_reach_weight_clip = min(self.reach_weight_clip, bounded_clip)

    def _compute_optimal_reach_clip(self):
        self._refresh_effective_reach_clip()
        if not self._reach_weight_observations:
            return {
                "reach_weight_clip_optimal": float(min(self.reach_weight_clip, 1.0)),
                "reach_weight_raw_p50": 0.0,
                "reach_weight_raw_p95": 0.0,
                "reach_weight_raw_p99": 0.0,
            }
        observed = np.asarray(self._reach_weight_observations, dtype=np.float64)
        return {
            "reach_weight_clip_optimal": float(self.effective_reach_weight_clip),
            "reach_weight_raw_p50": float(np.quantile(observed, 0.50)),
            "reach_weight_raw_p95": float(np.quantile(observed, 0.95)),
            "reach_weight_raw_p99": float(np.quantile(observed, 0.99)),
        }

    def _compute_reach_weights(self, reach_probs, current_player: int):
        if not self.use_reach_weighting:
            return 1.0, 1.0
        own_reach = float(max(1e-8, reach_probs[current_player]))
        opp_reach = 1.0
        for idx, rp in enumerate(reach_probs):
            if idx == current_player:
                continue
            opp_reach *= float(max(1e-8, rp))
        if self.reach_weight_mode == "sqrt":
            own_reach = float(np.sqrt(own_reach))
            opp_reach = float(np.sqrt(opp_reach))
        clip_value = max(1e-8, float(self.effective_reach_weight_clip))
        policy_weight = min(clip_value, own_reach)
        counterfactual_weight = min(clip_value, opp_reach)
        self._observe_reach_weight(policy_weight)
        self._observe_reach_weight(counterfactual_weight)
        return counterfactual_weight, policy_weight

    def _traverse(self, env, advantage_records, policy_records, stats, rng, reach_probs):
        state = env._get_state()
        if state.get("done", False) or state["street"] == "showdown":
            return self._terminal_utilities(env)

        player = state["current_player"]
        x = self.get_state_tensor(state, player)
        x_cpu = x.detach().cpu()
        legal_actions = env.get_legal_actions()

        with torch.inference_mode():
            pred_regrets = self.regret_net(x.unsqueeze(0)).squeeze(0).cpu().numpy()

        strategy = self._regret_matching(legal_actions, pred_regrets)
        counterfactual_weight, policy_weight = self._compute_reach_weights(reach_probs, player)
        policy_records.append((x_cpu, strategy.copy(), policy_weight))

        stats["visited_nodes"] += 1
        stats["legal_action_total"] += len(legal_actions)
        stats["entropy_total"] += self._strategy_entropy(strategy, legal_actions)
        stats["reach_cf_weight_total"] += float(counterfactual_weight)
        stats["reach_policy_weight_total"] += float(policy_weight)
        stats["reach_weight_count"] += 1

        chosen_idx, chosen_action = self._sample_action(strategy, legal_actions, rng)
        action_utility_vectors = np.zeros((self.num_actions, env.num_players), dtype=np.float32)

        chosen_env = env.clone()
        _, _, done, info = chosen_env.step(chosen_action)
        next_reach = np.array(reach_probs, dtype=np.float64, copy=True)
        next_reach[player] *= float(max(strategy[chosen_idx], 1e-8))
        if done:
            chosen_utility = self._terminal_utilities(chosen_env, info)
        else:
            chosen_utility = self._traverse(
                chosen_env,
                advantage_records,
                policy_records,
                stats,
                rng,
                next_reach,
            )
        action_utility_vectors[chosen_idx] = chosen_utility

        alternative_values = self._evaluate_other_actions(env, legal_actions, chosen_action)
        for action, utility_vector in alternative_values.items():
            action_utility_vectors[self.action_to_index[action]] = utility_vector

        action_player_values = action_utility_vectors[:, player]
        state_value = float(np.dot(strategy, action_player_values))

        for action in legal_actions:
            idx = self.action_to_index[action]
            regret = float(action_player_values[idx] - state_value)
            advantage_records.append((x_cpu, idx, regret, counterfactual_weight))
            stats["regret_abs_total"] += abs(regret)
            stats["regret_count"] += 1

        return chosen_utility

    def train_regret_net(self, epochs=None):
        if len(self.advantage_buffer) < max(64, self.batch_size):
            return 0.0

        dataset = AdvantageDataset(self.advantage_buffer)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(self.dataloader_workers > 0),
        )

        total_loss = 0.0
        epochs = self.regret_epochs if epochs is None else max(1, int(epochs))

        for _ in range(epochs):
            for x, a_idx, target_regret, sample_weight in loader:
                x = x.to(self.device)
                a_idx = a_idx.to(self.device)
                target_regret = target_regret.to(self.device)
                sample_weight = sample_weight.to(self.device)

                self.regret_opt.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    pred = self.regret_net(x)
                    self._require_finite_tensor(pred, "regret_net_output")
                    pred_val = pred.gather(1, a_idx.unsqueeze(1)).squeeze(1)
                    per_sample = torch.abs(pred_val - target_regret)
                    loss = self._weighted_mean(per_sample, sample_weight)
                self._require_finite_tensor(loss.unsqueeze(0), "regret_loss")

                self._scaler.scale(loss).backward()
                if self.grad_clip > 0.0:
                    self._scaler.unscale_(self.regret_opt)
                    clip_grad_norm_(self.regret_net.parameters(), self.grad_clip)
                self._scaler.step(self.regret_opt)
                self._scaler.update()
                loss_value = float(loss.item())
                self._require_finite_scalar(loss_value, "regret_loss_value")
                total_loss += loss_value

        return total_loss / max(1, len(loader) * epochs)

    def train_policy_net(self, epochs=None):
        if len(self.policy_buffer) < max(64, self.batch_size):
            return 0.0

        dataset = PolicyDataset(self.policy_buffer)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(self.dataloader_workers > 0),
        )

        total_loss = 0.0
        epochs = self.policy_epochs if epochs is None else max(1, int(epochs))

        for _ in range(epochs):
            for x, target_strat, sample_weight in loader:
                x = x.to(self.device)
                target_strat = target_strat.to(self.device)
                sample_weight = sample_weight.to(self.device)

                self.policy_opt.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    log_probs = self.policy_net.log_probs(x)
                    self._require_finite_tensor(log_probs, "policy_net_log_probs")
                    per_sample_kl = torch.sum(
                        target_strat * (torch.log(torch.clamp(target_strat, min=1e-8)) - log_probs),
                        dim=1,
                    )
                    loss = self._weighted_mean(per_sample_kl, sample_weight)
                    if self.entropy_regularization > 0.0:
                        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1).mean()
                        loss = loss - (self.entropy_regularization * entropy)
                self._require_finite_tensor(loss.unsqueeze(0), "policy_loss")

                self._scaler.scale(loss).backward()
                if self.grad_clip > 0.0:
                    self._scaler.unscale_(self.policy_opt)
                    clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
                self._scaler.step(self.policy_opt)
                self._scaler.update()
                loss_value = float(loss.item())
                self._require_finite_scalar(loss_value, "policy_loss_value")
                total_loss += loss_value

        return total_loss / max(1, len(loader) * epochs)

    def _play_hand(self, env, action_selector):
        state = env.reset()
        done = False
        final_info = {}

        while not done:
            player = state["current_player"]
            action = action_selector(state, player)
            state, _, done, final_info = env.step(action)

        return state, self._terminal_utilities(env, final_info), final_info

    def _action_from_policy_net(self, policy_net, env, state, player, rng):
        legal_actions = env.get_legal_actions()
        strategy = self._policy_strategy_from_net(policy_net, state, player, legal_actions)
        _, action = self._sample_action(strategy, legal_actions, rng)
        return action

    def _policy_action(self, env, state, player, rng):
        return self._action_from_policy_net(self.policy_net, env, state, player, rng)

    def _random_action(self, env, rng=None):
        legal_actions = env.get_legal_actions()
        active_rng = self.rng if rng is None else rng
        idx = int(active_rng.integers(len(legal_actions)))
        return legal_actions[idx]

    def _heuristic_action(self, env, state, player, style: str = "tag"):
        legal_actions = env.get_legal_actions()
        bet_actions = self._sorted_bet_actions(legal_actions)
        features = self.iss.encode(state, player)
        equity = float(features["equity"])
        pot_odds = float(features["pot_odds"])
        facing_bet = bool(features["facing_bet"])
        style = style.lower()
        thresholds = {
            "nit": {"fold_mult": 0.95, "bluff_shift": -0.08},
            "tag": {"fold_mult": 0.80, "bluff_shift": 0.00},
            "lag": {"fold_mult": 0.65, "bluff_shift": 0.08},
        }
        cfg = thresholds.get(style, thresholds["tag"])
        bluff_shift = cfg["bluff_shift"]

        if facing_bet:
            if equity < max(0.18, pot_odds * cfg["fold_mult"]) and "fold" in legal_actions:
                return "fold"
            if equity > 0.82 + bluff_shift and bet_actions:
                return bet_actions[-1]
            if equity > 0.70 + bluff_shift and len(bet_actions) >= 2:
                return bet_actions[-2]
            if equity >= pot_odds * 0.95 and "call" in legal_actions:
                return "call"
            return "fold" if "fold" in legal_actions else legal_actions[0]

        if equity > 0.88 + bluff_shift and "all_in" in legal_actions:
            return "all_in"
        if equity > 0.76 + bluff_shift and bet_actions:
            return bet_actions[-1]
        if equity > 0.64 + bluff_shift and len(bet_actions) >= 2:
            return bet_actions[-2]
        if equity > 0.52 + bluff_shift and bet_actions:
            return bet_actions[0]
        return "check" if "check" in legal_actions else legal_actions[0]

    def _collect_eval_metrics(self, hand_results, num_players: int):
        seat_utilities = np.zeros(num_players, dtype=np.float64)
        seat_win_shares = np.zeros(num_players, dtype=np.float64)
        seat_tie_rates = np.zeros(num_players, dtype=np.float64)
        showdown_hands = 0
        utility_sum_error = 0.0

        for final_state, utilities, info in hand_results:
            seat_utilities += utilities
            utility_sum_error += abs(float(np.sum(utilities)))
            winners = info.get("winners", [])
            if winners:
                win_share = 1.0 / len(winners)
                for winner in winners:
                    seat_win_shares[winner] += win_share
                    if len(winners) > 1:
                        seat_tie_rates[winner] += 1.0
            if final_state["street"] == "showdown":
                showdown_hands += 1

        num_hands = max(1, len(hand_results))
        seat_ev = seat_utilities / num_hands
        return {
            "seat_ev_bb_per_hand": seat_ev.tolist(),
            "seat_win_share": (seat_win_shares / num_hands).tolist(),
            "seat_tie_rate": (seat_tie_rates / num_hands).tolist(),
            "seat_ev_std": float(np.std(seat_ev)),
            "mean_abs_seat_ev_bb_per_hand": float(np.mean(np.abs(seat_ev))),
            "showdown_rate": showdown_hands / num_hands,
            "utility_sum_error": utility_sum_error / num_hands,
        }

    def evaluate_self_play(self, num_hands: int = 100, seed=None):
        master_rng = self._spawn_rng(seed)
        eval_env = self._make_isolated_env(seed=self._child_seed(master_rng))
        hand_results = []
        for _ in range(num_hands):
            local_rng = np.random.default_rng(self._child_seed(master_rng))
            hand_results.append(self._play_hand(eval_env, lambda state, player: self._policy_action(eval_env, state, player, local_rng)))
        return self._collect_eval_metrics(hand_results, num_players=eval_env.num_players)

    def evaluate_against_random(self, num_hands: int = 100, seed=None):
        master_rng = self._spawn_rng(seed)
        eval_env = self._make_isolated_env(seed=self._child_seed(master_rng))
        hero_total_utility = 0.0
        hero_win_share = 0.0
        hero_tie_rate = 0.0
        showdown_hands = 0
        seat_totals = np.zeros(eval_env.num_players, dtype=np.float64)
        seat_counts = np.zeros(eval_env.num_players, dtype=np.float64)
        hero_utilities = []

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % eval_env.num_players
            seat_counts[hero_seat] += 1.0
            local_rng = np.random.default_rng(self._child_seed(master_rng))

            def action_selector(state, player):
                if player == hero_seat:
                    return self._policy_action(eval_env, state, player, local_rng)
                return self._random_action(eval_env, local_rng)

            final_state, utilities, info = self._play_hand(eval_env, action_selector)
            hero_utility = float(utilities[hero_seat])
            hero_total_utility += hero_utility
            seat_totals[hero_seat] += hero_utility
            hero_utilities.append(hero_utility)

            winners = info.get("winners", [])
            if hero_seat in winners:
                hero_win_share += 1.0 / len(winners)
                if len(winners) > 1:
                    hero_tie_rate += 1.0

            if final_state["street"] == "showdown":
                showdown_hands += 1

        seat_ev = []
        for seat in range(eval_env.num_players):
            if seat_counts[seat] > 0:
                seat_ev.append(float(seat_totals[seat] / seat_counts[seat]))
            else:
                seat_ev.append(0.0)

        hero_ev = hero_total_utility / max(1, num_hands)
        hero_std = float(np.std(hero_utilities)) if hero_utilities else 0.0
        hero_stderr = hero_std / max(1.0, float(np.sqrt(max(1, len(hero_utilities)))))
        return {
            "hero_ev_bb_per_hand": hero_ev,
            "hero_bb_per_100": hero_ev * 100.0,
            "hero_bb_per_100_stderr": hero_stderr * 100.0,
            "hero_win_share": hero_win_share / max(1, num_hands),
            "hero_tie_rate": hero_tie_rate / max(1, num_hands),
            "showdown_rate": showdown_hands / max(1, num_hands),
            "hero_seat_ev_bb_per_hand": seat_ev,
        }

    def evaluate_against_heuristic(self, num_hands: int = 100, seed=None):
        master_rng = self._spawn_rng(seed)
        eval_env = self._make_isolated_env(seed=self._child_seed(master_rng))
        hero_total_utility = 0.0
        hero_win_share = 0.0
        hero_tie_rate = 0.0
        showdown_hands = 0
        hero_utilities = []

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % eval_env.num_players
            local_rng = np.random.default_rng(self._child_seed(master_rng))

            def action_selector(state, player):
                if player == hero_seat:
                    return self._policy_action(eval_env, state, player, local_rng)
                return self._heuristic_action(eval_env, state, player, style="tag")

            final_state, utilities, info = self._play_hand(eval_env, action_selector)
            hero_total_utility += float(utilities[hero_seat])
            hero_utilities.append(float(utilities[hero_seat]))

            winners = info.get("winners", [])
            if hero_seat in winners:
                hero_win_share += 1.0 / len(winners)
                if len(winners) > 1:
                    hero_tie_rate += 1.0

            if final_state["street"] == "showdown":
                showdown_hands += 1

        hero_ev = hero_total_utility / max(1, num_hands)
        hero_std = float(np.std(hero_utilities)) if hero_utilities else 0.0
        hero_stderr = hero_std / max(1.0, float(np.sqrt(max(1, len(hero_utilities)))))
        return {
            "hero_ev_bb_per_hand": hero_ev,
            "hero_bb_per_100": hero_ev * 100.0,
            "hero_bb_per_100_stderr": hero_stderr * 100.0,
            "hero_win_share": hero_win_share / max(1, num_hands),
            "hero_tie_rate": hero_tie_rate / max(1, num_hands),
            "showdown_rate": showdown_hands / max(1, num_hands),
        }

    def evaluate_against_heuristic_pool(self, num_hands: int = 100, seed=None):
        master_rng = self._spawn_rng(seed)
        eval_env = self._make_isolated_env(seed=self._child_seed(master_rng))
        styles = ("nit", "tag", "lag")
        hero_total_utility = 0.0
        showdown_hands = 0
        hero_utilities = []

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % eval_env.num_players
            local_rng = np.random.default_rng(self._child_seed(master_rng))

            def action_selector(state, player):
                if player == hero_seat:
                    return self._policy_action(eval_env, state, player, local_rng)
                style = styles[(hand_idx + player) % len(styles)]
                return self._heuristic_action(eval_env, state, player, style=style)

            final_state, utilities, _ = self._play_hand(eval_env, action_selector)
            hero_total_utility += float(utilities[hero_seat])
            hero_utilities.append(float(utilities[hero_seat]))
            if final_state["street"] == "showdown":
                showdown_hands += 1

        hero_ev = hero_total_utility / max(1, num_hands)
        hero_std = float(np.std(hero_utilities)) if hero_utilities else 0.0
        hero_stderr = hero_std / max(1.0, float(np.sqrt(max(1, len(hero_utilities)))))
        return {
            "hero_ev_bb_per_hand": hero_ev,
            "hero_bb_per_100": hero_ev * 100.0,
            "hero_bb_per_100_stderr": hero_stderr * 100.0,
            "showdown_rate": showdown_hands / max(1, num_hands),
            "pool_profiles": list(styles),
        }

    def _load_snapshot_policy(self, snapshot_path):
        cache_key = str(snapshot_path)
        with self._snapshot_cache_lock:
            cached_model = self.snapshot_policy_cache.get(cache_key)
            if cached_model is not None:
                self.snapshot_policy_cache.move_to_end(cache_key)
                return cached_model

        payload = torch.load(snapshot_path, map_location=self.device, weights_only=False)
        model = PolicyNet.from_checkpoint_payload(
            payload,
            input_dim=self.input_dim,
            output_dim=self.num_actions,
            device=self.device,
        )
        model.eval()

        with self._snapshot_cache_lock:
            cached_model = self.snapshot_policy_cache.get(cache_key)
            if cached_model is not None:
                self.snapshot_policy_cache.move_to_end(cache_key)
                return cached_model

            self.snapshot_policy_cache[cache_key] = model
            self.snapshot_policy_cache.move_to_end(cache_key)
            while len(self.snapshot_policy_cache) > self.max_snapshot_cache:
                self.snapshot_policy_cache.popitem(last=False)
        return model

    def evaluate_against_snapshot_pool(self, num_hands: int = 100, before_iteration=None, seed=None):
        if self.checkpoint_manager is None or self.snapshot_pool_size <= 0:
            return None

        snapshot_paths = self.checkpoint_manager.list_snapshot_paths(
            before_iteration=before_iteration,
            limit=self.snapshot_pool_size,
        )
        if not snapshot_paths:
            return None

        snapshot_models = [self._load_snapshot_policy(path) for path in snapshot_paths]
        master_rng = self._spawn_rng(seed)
        eval_env = self._make_isolated_env(seed=self._child_seed(master_rng))
        hero_total_utility = 0.0
        hero_win_share = 0.0
        hero_tie_rate = 0.0
        showdown_hands = 0
        hero_utilities = []

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % eval_env.num_players
            local_rng = np.random.default_rng(self._child_seed(master_rng))

            seat_models = {}
            snapshot_index = 0
            for seat in range(eval_env.num_players):
                if seat == hero_seat:
                    continue
                seat_models[seat] = snapshot_models[(hand_idx + snapshot_index) % len(snapshot_models)]
                snapshot_index += 1

            def action_selector(state, player):
                if player == hero_seat:
                    return self._policy_action(eval_env, state, player, local_rng)
                return self._action_from_policy_net(seat_models[player], eval_env, state, player, local_rng)

            final_state, utilities, info = self._play_hand(eval_env, action_selector)
            hero_total_utility += float(utilities[hero_seat])
            hero_utilities.append(float(utilities[hero_seat]))

            winners = info.get("winners", [])
            if hero_seat in winners:
                hero_win_share += 1.0 / len(winners)
                if len(winners) > 1:
                    hero_tie_rate += 1.0

            if final_state["street"] == "showdown":
                showdown_hands += 1

        hero_ev = hero_total_utility / max(1, num_hands)
        hero_std = float(np.std(hero_utilities)) if hero_utilities else 0.0
        hero_stderr = hero_std / max(1.0, float(np.sqrt(max(1, len(hero_utilities)))))
        return {
            "hero_ev_bb_per_hand": hero_ev,
            "hero_bb_per_100": hero_ev * 100.0,
            "hero_bb_per_100_stderr": hero_stderr * 100.0,
            "hero_win_share": hero_win_share / max(1, num_hands),
            "hero_tie_rate": hero_tie_rate / max(1, num_hands),
            "showdown_rate": showdown_hands / max(1, num_hands),
            "pool_size": len(snapshot_models),
        }

    def evaluate_against_population(self, num_hands: int = 100, before_iteration=None, seed=None):
        population_paths = self._combined_population_paths(before_iteration=before_iteration)
        if not population_paths:
            return None

        population_models = [self._load_snapshot_policy(path) for path in population_paths]
        master_rng = self._spawn_rng(seed)
        eval_env = self._make_isolated_env(seed=self._child_seed(master_rng))
        hero_total_utility = 0.0
        hero_win_share = 0.0
        hero_tie_rate = 0.0
        showdown_hands = 0
        hero_utilities = []

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % eval_env.num_players
            local_rng = np.random.default_rng(self._child_seed(master_rng))
            seat_models = {}

            for seat in range(eval_env.num_players):
                if seat == hero_seat:
                    continue
                seat_models[seat] = population_models[(hand_idx + seat) % len(population_models)]

            def action_selector(state, player):
                if player == hero_seat:
                    return self._policy_action(eval_env, state, player, local_rng)
                return self._action_from_policy_net(seat_models[player], eval_env, state, player, local_rng)

            final_state, utilities, info = self._play_hand(eval_env, action_selector)
            hero_total_utility += float(utilities[hero_seat])
            hero_utilities.append(float(utilities[hero_seat]))

            winners = info.get("winners", [])
            if hero_seat in winners:
                hero_win_share += 1.0 / len(winners)
                if len(winners) > 1:
                    hero_tie_rate += 1.0

            if final_state["street"] == "showdown":
                showdown_hands += 1

        hero_ev = hero_total_utility / max(1, num_hands)
        hero_std = float(np.std(hero_utilities)) if hero_utilities else 0.0
        hero_stderr = hero_std / max(1.0, float(np.sqrt(max(1, len(hero_utilities)))))
        return {
            "hero_ev_bb_per_hand": hero_ev,
            "hero_bb_per_100": hero_ev * 100.0,
            "hero_bb_per_100_stderr": hero_stderr * 100.0,
            "hero_win_share": hero_win_share / max(1, num_hands),
            "hero_tie_rate": hero_tie_rate / max(1, num_hands),
            "showdown_rate": showdown_hands / max(1, num_hands),
            "pool_size": len(population_models),
        }

    def evaluate_ev(self, num_hands: int = 100):
        metrics = self.evaluate_against_random(num_hands=num_hands)
        return (
            metrics["hero_ev_bb_per_hand"],
            metrics["hero_win_share"],
            metrics["hero_tie_rate"],
            metrics["hero_bb_per_100"],
        )

    def evaluate_best_response_proxy(self, num_hands: int = 50, seed=None):
        master_rng = self._spawn_rng(seed)
        eval_env = self._make_isolated_env(seed=self._child_seed(master_rng))
        total_br_utility = 0.0
        total_policy_utility = 0.0
        hand_count = max(1, int(num_hands))

        for hand_idx in range(hand_count):
            hero_seat = hand_idx % eval_env.num_players
            br_env = eval_env.clone()
            policy_env = eval_env.clone()
            br_env.reset()
            policy_env.reset()
            local_rng = np.random.default_rng(self._child_seed(master_rng))

            # Approximate BR: hero greedily picks the action with best rollout EV at each decision.
            br_done = False
            br_info = {}
            while not br_done:
                br_state = br_env._get_state()
                player = br_state["current_player"]
                if player == hero_seat:
                    legal_actions = br_env.get_legal_actions()
                    best_action = legal_actions[0]
                    best_value = float("-inf")
                    for action in legal_actions:
                        value = float(self._estimate_action_utility(br_env, action, self._child_seed(master_rng))[hero_seat])
                        if value > best_value:
                            best_value = value
                            best_action = action
                    _, _, br_done, br_info = br_env.step(best_action)
                else:
                    action = self._policy_action(br_env, br_state, player, local_rng)
                    _, _, br_done, br_info = br_env.step(action)

            # Baseline policy-vs-policy with same seat rotation.
            pol_done = False
            pol_info = {}
            while not pol_done:
                pol_state = policy_env._get_state()
                player = pol_state["current_player"]
                action = self._policy_action(policy_env, pol_state, player, local_rng)
                _, _, pol_done, pol_info = policy_env.step(action)

            total_br_utility += float(self._terminal_utilities(br_env, br_info)[hero_seat])
            total_policy_utility += float(self._terminal_utilities(policy_env, pol_info)[hero_seat])

        br_ev = total_br_utility / hand_count
        policy_ev = total_policy_utility / hand_count
        return {
            "br_ev_bb_per_hand": br_ev,
            "policy_ev_bb_per_hand": policy_ev,
            "br_gap_proxy": max(0.0, br_ev - policy_ev),
            "br_gap_proxy_bb_per_100": max(0.0, br_ev - policy_ev) * 100.0,
        }

    def _save_checkpoint(self, iteration_number, metrics):
        if self.checkpoint_manager is None:
            return

        is_best = metrics["vs_random_bb_per_100"] >= self.best_vs_random_bb_per_100
        if is_best:
            self.best_vs_random_bb_per_100 = metrics["vs_random_bb_per_100"]
        history = [entry.get("robust_score") for entry in self.metrics_history if entry.get("robust_score") is not None]
        best_robust_so_far = max(history) if history else float("-inf")
        robust_score = metrics.get("robust_score")
        is_best_robust = robust_score is not None and robust_score >= best_robust_so_far

        if iteration_number % self.checkpoint_interval != 0 and not is_best and not is_best_robust:
            return

        # Flush any pending LUT writes before we commit the checkpoint
        try:
            self.iss.card_abs.equity_provider.flush()
        except Exception:
            pass  # non-critical

        payload = self.state_dict()
        payload["last_metrics"] = metrics
        self.checkpoint_manager.save_checkpoint(
            payload,
            iteration_number,
            metrics=metrics,
            is_best=is_best,
            is_best_robust=is_best_robust,
        )

    @staticmethod
    def _risk_adjusted_score(bb_per_100, bb_per_100_stderr):
        if bb_per_100 is None:
            return None
        penalty = 0.25 * float(bb_per_100_stderr or 0.0)
        return float(bb_per_100) - penalty

    def _compute_robust_score(self, metrics):
        weighted_terms = []
        weighting = (
            ("vs_random_bb_per_100", "vs_random_bb_per_100_stderr", 0.35),
            ("vs_snapshot_bb_per_100", "vs_snapshot_bb_per_100_stderr", 0.20),
            ("vs_population_bb_per_100", "vs_population_bb_per_100_stderr", 0.20),
            ("vs_heuristic_bb_per_100", "vs_heuristic_bb_per_100_stderr", 0.15),
            ("vs_heuristic_pool_bb_per_100", "vs_heuristic_pool_bb_per_100_stderr", 0.10),
        )

        for mean_key, stderr_key, weight in weighting:
            adjusted = self._risk_adjusted_score(metrics.get(mean_key), metrics.get(stderr_key))
            if adjusted is None:
                continue
            weighted_terms.append((adjusted, weight))

        if not weighted_terms:
            return None

        total_weight = sum(weight for _, weight in weighted_terms)
        if total_weight <= 0:
            return None

        score = sum(adjusted * weight for adjusted, weight in weighted_terms) / total_weight
        return float(score)

    @staticmethod
    def _compute_exploitability_proxy(metrics):
        return float(
            abs(float(metrics.get("self_play_mean_abs_seat_ev", 0.0))) +
            abs(float(metrics.get("self_play_utility_sum_error", 0.0))) +
            abs(float(metrics.get("avg_abs_regret", 0.0))) * 0.1
        )

    def train(
        self,
        iterations: int = 10,
        episodes: int = 20,
        eval_hands: int = 100,
        eval_random_hands: int = 0,
        eval_snapshot_hands: int = 0,
        eval_population_hands: int = 0,
        eval_heuristic_hands: int = 0,
        early_stop_patience: int = 0,
        early_stop_min_iters: int = 5,
        early_stop_entropy_floor: float = 0.02,
        early_stop_entropy_patience: int = 3,
        early_stop_regret_loss_ceiling: float = 500.0,
        early_stop_policy_loss_ceiling: float = 25.0,
    ):
        random_eval_hands = eval_random_hands if eval_random_hands > 0 else max(50, eval_hands // 2)
        snapshot_eval_hands = eval_snapshot_hands if eval_snapshot_hands > 0 else random_eval_hands
        population_eval_hands = eval_population_hands if eval_population_hands > 0 else random_eval_hands
        heuristic_eval_hands = eval_heuristic_hands if eval_heuristic_hands > 0 else random_eval_hands

        no_improve_count = 0
        low_entropy_count = 0
        best_robust_score = float("-inf")

        for i in range(self.completed_iterations, iterations):
            iteration_number = i + 1
            iter_t0 = time.monotonic()
            self._refresh_effective_reach_clip()
            print(f"[DeepCFR] Iter {iteration_number:03d}/{iterations}: starting self-play ({episodes} episodes)")
            self.iss.card_abs.reset_equity_stats()
            self._refresh_population_pool()
            sp_t0 = time.monotonic()
            traversal_stats = self.self_play(episodes=episodes)
            sp_wall = time.monotonic() - sp_t0
            print(f"[DeepCFR] Iter {iteration_number:03d}: training regret net")
            r_loss = self.train_regret_net()
            print(f"[DeepCFR] Iter {iteration_number:03d}: training policy net")
            p_loss = self.train_policy_net()
            eval_t0 = time.monotonic()
            print(f"[DeepCFR] Iter {iteration_number:03d}: evaluating (parallel)")
            prev_policy_mode = self.policy_net.training
            prev_regret_mode = self.regret_net.training
            self.policy_net.eval()
            self.regret_net.eval()
            try:
                eval_seed_rng = self._spawn_rng()
                with ThreadPoolExecutor(max_workers=7) as eval_executor:
                    fut_self_play = eval_executor.submit(
                        self.evaluate_self_play,
                        num_hands=eval_hands,
                        seed=self._child_seed(eval_seed_rng),
                    )
                    fut_vs_random = eval_executor.submit(
                        self.evaluate_against_random,
                        num_hands=random_eval_hands,
                        seed=self._child_seed(eval_seed_rng),
                    )
                    fut_heuristic = eval_executor.submit(
                        self.evaluate_against_heuristic,
                        num_hands=heuristic_eval_hands,
                        seed=self._child_seed(eval_seed_rng),
                    )
                    fut_heuristic_pool = eval_executor.submit(
                        self.evaluate_against_heuristic_pool,
                        num_hands=heuristic_eval_hands,
                        seed=self._child_seed(eval_seed_rng),
                    )
                    fut_br_proxy = eval_executor.submit(
                        self.evaluate_best_response_proxy,
                        num_hands=max(20, random_eval_hands // 2),
                        seed=self._child_seed(eval_seed_rng),
                    )
                    fut_snapshot = eval_executor.submit(
                        self.evaluate_against_snapshot_pool,
                        num_hands=snapshot_eval_hands,
                        before_iteration=iteration_number,
                        seed=self._child_seed(eval_seed_rng),
                    )
                    fut_population = eval_executor.submit(
                        self.evaluate_against_population,
                        num_hands=population_eval_hands,
                        before_iteration=iteration_number,
                        seed=self._child_seed(eval_seed_rng),
                    )
                    self_play_metrics = fut_self_play.result()
                    vs_random_metrics = fut_vs_random.result()
                    heuristic_metrics = fut_heuristic.result()
                    heuristic_pool_metrics = fut_heuristic_pool.result()
                    br_proxy_metrics = fut_br_proxy.result()
                    snapshot_metrics = fut_snapshot.result()
                    population_metrics = fut_population.result()
            finally:
                self.policy_net.train(prev_policy_mode)
                self.regret_net.train(prev_regret_mode)
            eval_wall = time.monotonic() - eval_t0
            iter_wall = time.monotonic() - iter_t0

            metrics = {
                "regret_loss": r_loss,
                "policy_loss": p_loss,
                "iteration_wall_time_seconds": iter_wall,
                "self_play_wall_time_seconds": sp_wall,
                "evaluation_wall_time_seconds": eval_wall,
                "traversal_nodes": traversal_stats["visited_nodes"],
                "avg_branching_factor": traversal_stats["avg_branching_factor"],
                "avg_policy_entropy": traversal_stats["avg_policy_entropy"],
                "avg_abs_regret": traversal_stats["avg_abs_regret"],
                "reach_cf_weight_mean": traversal_stats["reach_cf_weight_mean"],
                "reach_policy_weight_mean": traversal_stats["reach_policy_weight_mean"],
                "effective_reach_weight_clip": float(self.effective_reach_weight_clip),
                "self_play_seat_ev_std": self_play_metrics["seat_ev_std"],
                "self_play_mean_abs_seat_ev": self_play_metrics["mean_abs_seat_ev_bb_per_hand"],
                "self_play_showdown_rate": self_play_metrics["showdown_rate"],
                "self_play_utility_sum_error": self_play_metrics["utility_sum_error"],
                "vs_random_ev": vs_random_metrics["hero_ev_bb_per_hand"],
                "vs_random_win_share": vs_random_metrics["hero_win_share"],
                "vs_random_tie_rate": vs_random_metrics["hero_tie_rate"],
                "vs_random_bb_per_100": vs_random_metrics["hero_bb_per_100"],
                "vs_random_showdown_rate": vs_random_metrics["showdown_rate"],
                "vs_random_bb_per_100_stderr": vs_random_metrics["hero_bb_per_100_stderr"],
                "vs_heuristic_ev": heuristic_metrics["hero_ev_bb_per_hand"],
                "vs_heuristic_win_share": heuristic_metrics["hero_win_share"],
                "vs_heuristic_tie_rate": heuristic_metrics["hero_tie_rate"],
                "vs_heuristic_bb_per_100": heuristic_metrics["hero_bb_per_100"],
                "vs_heuristic_showdown_rate": heuristic_metrics["showdown_rate"],
                "vs_heuristic_bb_per_100_stderr": heuristic_metrics["hero_bb_per_100_stderr"],
                "vs_heuristic_pool_ev": heuristic_pool_metrics["hero_ev_bb_per_hand"],
                "vs_heuristic_pool_bb_per_100": heuristic_pool_metrics["hero_bb_per_100"],
                "vs_heuristic_pool_bb_per_100_stderr": heuristic_pool_metrics["hero_bb_per_100_stderr"],
                "vs_heuristic_pool_showdown_rate": heuristic_pool_metrics["showdown_rate"],
                "br_ev_bb_per_hand": br_proxy_metrics["br_ev_bb_per_hand"],
                "policy_ev_bb_per_hand": br_proxy_metrics["policy_ev_bb_per_hand"],
                "br_gap_proxy": br_proxy_metrics["br_gap_proxy"],
                "br_gap_proxy_bb_per_100": br_proxy_metrics["br_gap_proxy_bb_per_100"],
            }

            if snapshot_metrics is not None:
                metrics["vs_snapshot_ev"] = snapshot_metrics["hero_ev_bb_per_hand"]
                metrics["vs_snapshot_win_share"] = snapshot_metrics["hero_win_share"]
                metrics["vs_snapshot_tie_rate"] = snapshot_metrics["hero_tie_rate"]
                metrics["vs_snapshot_bb_per_100"] = snapshot_metrics["hero_bb_per_100"]
                metrics["vs_snapshot_bb_per_100_stderr"] = snapshot_metrics["hero_bb_per_100_stderr"]
                metrics["vs_snapshot_showdown_rate"] = snapshot_metrics["showdown_rate"]
                metrics["snapshot_pool_size"] = snapshot_metrics["pool_size"]

            if population_metrics is not None:
                metrics["vs_population_ev"] = population_metrics["hero_ev_bb_per_hand"]
                metrics["vs_population_win_share"] = population_metrics["hero_win_share"]
                metrics["vs_population_tie_rate"] = population_metrics["hero_tie_rate"]
                metrics["vs_population_bb_per_100"] = population_metrics["hero_bb_per_100"]
                metrics["vs_population_bb_per_100_stderr"] = population_metrics["hero_bb_per_100_stderr"]
                metrics["vs_population_showdown_rate"] = population_metrics["showdown_rate"]
                metrics["population_pool_size"] = population_metrics["pool_size"]

            metrics.update(self.iss.card_abs.get_equity_stats())
            metrics.update(self.iss.get_cache_stats())
            metrics.update(self._compute_optimal_reach_clip())
            metrics["robust_score"] = self._compute_robust_score(metrics)
            metrics["exploitability_proxy"] = self._compute_exploitability_proxy(metrics)

            robust_score = metrics.get("robust_score")
            if robust_score is not None:
                if robust_score > best_robust_score:
                    best_robust_score = robust_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if traversal_stats["avg_policy_entropy"] < float(early_stop_entropy_floor):
                low_entropy_count += 1
            else:
                low_entropy_count = 0

            snapshot_str = "n/a"
            if snapshot_metrics is not None:
                snapshot_str = f"{snapshot_metrics['hero_bb_per_100']:.2f}"

            population_str = "n/a"
            if population_metrics is not None:
                population_str = f"{population_metrics['hero_bb_per_100']:.2f}"

            print(
                f"[DeepCFR] Iter {iteration_number:03d} | Nodes={traversal_stats['visited_nodes']} | "
                f"Branch={traversal_stats['avg_branching_factor']:.2f} | "
                f"Entropy={traversal_stats['avg_policy_entropy']:.3f} | "
                f"SeatStd={self_play_metrics['seat_ev_std']:.4f} | "
                f"Showdown={self_play_metrics['showdown_rate']:.3f} | "
                f"VsRndBB/100={vs_random_metrics['hero_bb_per_100']:.2f} | "
                f"VsHeurBB/100={heuristic_metrics['hero_bb_per_100']:.2f} | "
                f"VsPoolBB/100={snapshot_str} | "
                f"VsPopBB/100={population_str} | "
                f"R-loss={r_loss:.6f} | P-loss={p_loss:.6f} | "
                f"Robust={metrics['robust_score'] if metrics['robust_score'] is not None else float('nan'):.2f}"
            )

            self.completed_iterations = iteration_number
            self.metrics_history.append(self._compact_metrics_entry(iteration_number, metrics))
            self._save_checkpoint(iteration_number, metrics)

            all_metrics = dict(metrics)
            all_metrics["iteration"] = float(iteration_number)
            for metric_name in self.IMPORTANT_METRIC_KEYS:
                metric_value = metrics.get(metric_name)
                if metric_value is not None:
                    all_metrics[f"important/{metric_name}"] = metric_value
            for seat, seat_ev in enumerate(self_play_metrics["seat_ev_bb_per_hand"]):
                all_metrics[f"self_play_seat_{seat}_ev"] = seat_ev
            for seat, seat_ev in enumerate(vs_random_metrics["hero_seat_ev_bb_per_hand"]):
                all_metrics[f"vs_random_seat_{seat}_ev"] = seat_ev
            log_metrics_batch(all_metrics, step=iteration_number)

            reached_min_iters = iteration_number >= max(1, int(early_stop_min_iters))
            if reached_min_iters:
                if r_loss > float(early_stop_regret_loss_ceiling):
                    print(f"[DeepCFR] Early stop: regret_loss {r_loss:.4f} exceeded ceiling {early_stop_regret_loss_ceiling:.4f}")
                    break
                if p_loss > float(early_stop_policy_loss_ceiling):
                    print(f"[DeepCFR] Early stop: policy_loss {p_loss:.4f} exceeded ceiling {early_stop_policy_loss_ceiling:.4f}")
                    break
                if int(early_stop_patience) > 0 and no_improve_count >= int(early_stop_patience):
                    print(f"[DeepCFR] Early stop: robust_score plateau for {no_improve_count} iterations (patience={early_stop_patience})")
                    break
                if int(early_stop_entropy_patience) > 0 and low_entropy_count >= int(early_stop_entropy_patience):
                    print(
                        f"[DeepCFR] Early stop: avg_policy_entropy below floor {early_stop_entropy_floor:.4f} "
                        f"for {low_entropy_count} consecutive iterations"
                    )
                    break
