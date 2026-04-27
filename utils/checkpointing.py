import json
from datetime import datetime
from pathlib import Path

import torch


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _make_run_name(seed=None):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if seed is None:
        return f"run_{timestamp}"
    return f"run_{timestamp}_seed{seed}"


class CheckpointManager:
    def __init__(
        self,
        root_dir: str = "artifacts/checkpoints",
        experiment: str = "poker_cfr_ai",
        run_name: str = None,
        resume_mode: str = "auto",
        seed=None,
        keep_last: int = 0,
    ):
        self.root_dir = Path(root_dir)
        self.experiment = experiment
        self.run_name = run_name
        self.resume_mode = resume_mode
        self.seed = seed
        self.keep_last = keep_last
        self.experiment_dir = self.root_dir / experiment
        self.run_dir = None
        self.run_summary_path = None
        self.latest_checkpoint_path = None
        self.best_checkpoint_path = None
        self.best_robust_checkpoint_path = None
        self.resume_checkpoint_path = None

    def prepare_run(self):
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if self.run_name is not None:
            candidate_run_dir = self.experiment_dir / self.run_name
            candidate_latest = candidate_run_dir / "latest.pt"
            if self.resume_mode != "never" and candidate_latest.exists():
                self.run_dir = candidate_run_dir
                self.resume_checkpoint_path = candidate_latest
            else:
                self.run_dir = candidate_run_dir
        else:
            pointer = self._read_latest_pointer()
            if self.resume_mode != "never" and pointer is not None:
                candidate_run_dir = Path(pointer["run_dir"])
                candidate_latest = candidate_run_dir / "latest.pt"
                if candidate_latest.exists():
                    self.run_dir = candidate_run_dir
                    self.run_name = pointer["run_name"]
                    self.resume_checkpoint_path = candidate_latest

            if self.run_dir is None:
                self.run_name = _make_run_name(seed=self.seed)
                self.run_dir = self._unique_run_dir(self.run_name)
                self.run_name = self.run_dir.name

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_summary_path = self.run_dir / "run_summary.json"
        self.latest_checkpoint_path = self.run_dir / "latest.pt"
        self.best_checkpoint_path = self.run_dir / "best.pt"
        self.best_robust_checkpoint_path = self.run_dir / "best_robust.pt"

        self._ensure_summary()
        self._write_latest_pointer()

        return {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "resume_checkpoint": str(self.resume_checkpoint_path) if self.resume_checkpoint_path is not None else None,
            "is_resumed": self.resume_checkpoint_path is not None,
        }

    def _unique_run_dir(self, base_run_name):
        candidate = self.experiment_dir / base_run_name
        if not candidate.exists():
            return candidate

        suffix = 1
        while True:
            candidate = self.experiment_dir / f"{base_run_name}_{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def load_resume_checkpoint(self, map_location="cpu"):
        if self.resume_checkpoint_path is None or not self.resume_checkpoint_path.exists():
            return None
        return torch.load(self.resume_checkpoint_path, map_location=map_location, weights_only=False)

    def save_checkpoint(self, payload, iteration: int, metrics=None, is_best: bool = False, is_best_robust: bool = False):
        checkpoint_path = self.run_dir / f"iter_{iteration:04d}.pt"
        torch.save(payload, checkpoint_path)
        torch.save(payload, self.latest_checkpoint_path)
        if is_best:
            torch.save(payload, self.best_checkpoint_path)
        if is_best_robust:
            torch.save(payload, self.best_robust_checkpoint_path)
        self._update_summary(iteration, metrics or {}, checkpoint_path, is_best=is_best, is_best_robust=is_best_robust)
        self._write_latest_pointer()
        self._cleanup_old_checkpoints()
        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        if self.keep_last <= 0:
            return
        protected = {
            self.latest_checkpoint_path,
            self.best_checkpoint_path,
            self.best_robust_checkpoint_path,
        }
        iter_files = sorted(self.run_dir.glob("iter_*.pt"))
        if len(iter_files) <= self.keep_last:
            return
        for old in iter_files[:-self.keep_last]:
            if old.resolve() not in {p.resolve() for p in protected if p is not None and p.exists()}:
                old.unlink(missing_ok=True)

    def list_snapshot_paths(self, before_iteration=None, limit=None):
        checkpoint_paths = sorted(self.run_dir.glob("iter_*.pt"))
        snapshots = []

        for path in checkpoint_paths:
            try:
                iteration = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if before_iteration is not None and iteration >= before_iteration:
                continue
            snapshots.append((iteration, path))

        if limit is not None:
            snapshots = snapshots[-limit:]

        return [path for _, path in snapshots]

    def _pointer_path(self):
        return self.experiment_dir / "latest_run.json"

    def _read_latest_pointer(self):
        pointer_path = self._pointer_path()
        if not pointer_path.exists():
            return None
        with pointer_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_latest_pointer(self):
        pointer_path = self._pointer_path()
        payload = {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "updated_at": _utc_now(),
        }
        with pointer_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def _ensure_summary(self):
        if self.run_summary_path.exists():
            return

        payload = {
            "experiment": self.experiment,
            "run_name": self.run_name,
            "seed": self.seed,
            "mlflow_run_id": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "latest_iteration": -1,
            "best_iteration": None,
            "best_vs_random_bb_per_100": None,
            "best_vs_snapshot_bb_per_100": None,
            "best_vs_population_bb_per_100": None,
            "best_vs_heuristic_bb_per_100": None,
            "best_robust_score": None,
            "best_robust_iteration": None,
            "checkpoints": [],
        }
        with self.run_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def _read_summary(self):
        with self.run_summary_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_mlflow_run_id(self):
        if self.run_summary_path is None or not self.run_summary_path.exists():
            return None
        summary = self._read_summary()
        return summary.get("mlflow_run_id")

    def set_mlflow_run_id(self, run_id: str):
        if self.run_summary_path is None:
            return
        summary = self._read_summary()
        summary["mlflow_run_id"] = run_id
        self._write_summary(summary)

    def _write_summary(self, payload):
        payload["updated_at"] = _utc_now()
        with self.run_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def _update_summary(self, iteration, metrics, checkpoint_path, is_best=False, is_best_robust=False):
        summary = self._read_summary()
        summary["latest_iteration"] = max(int(summary.get("latest_iteration", -1)), int(iteration))

        checkpoints = [entry for entry in summary.get("checkpoints", []) if entry.get("iteration") != iteration]
        checkpoints.append(
            {
                "iteration": int(iteration),
                "path": str(checkpoint_path),
                "metrics": metrics,
                "saved_at": _utc_now(),
            }
        )
        checkpoints.sort(key=lambda entry: entry["iteration"])
        summary["checkpoints"] = checkpoints

        vs_random = metrics.get("vs_random_bb_per_100")
        if vs_random is not None:
            current_best = summary.get("best_vs_random_bb_per_100")
            if current_best is None or vs_random >= current_best:
                summary["best_vs_random_bb_per_100"] = vs_random
                summary["best_iteration"] = int(iteration)

        vs_snapshot = metrics.get("vs_snapshot_bb_per_100")
        if vs_snapshot is not None:
            current_best_snapshot = summary.get("best_vs_snapshot_bb_per_100")
            if current_best_snapshot is None or vs_snapshot >= current_best_snapshot:
                summary["best_vs_snapshot_bb_per_100"] = vs_snapshot

        vs_population = metrics.get("vs_population_bb_per_100")
        if vs_population is not None:
            current_best_population = summary.get("best_vs_population_bb_per_100")
            if current_best_population is None or vs_population >= current_best_population:
                summary["best_vs_population_bb_per_100"] = vs_population

        vs_heuristic = metrics.get("vs_heuristic_bb_per_100")
        if vs_heuristic is not None:
            current_best_heuristic = summary.get("best_vs_heuristic_bb_per_100")
            if current_best_heuristic is None or vs_heuristic >= current_best_heuristic:
                summary["best_vs_heuristic_bb_per_100"] = vs_heuristic

        if is_best and summary.get("best_iteration") is None:
            summary["best_iteration"] = int(iteration)

        robust_score = metrics.get("robust_score")
        if robust_score is not None:
            current_best_robust = summary.get("best_robust_score")
            if current_best_robust is None or robust_score >= current_best_robust:
                summary["best_robust_score"] = robust_score
                summary["best_robust_iteration"] = int(iteration)

        if is_best_robust and summary.get("best_robust_iteration") is None:
            summary["best_robust_iteration"] = int(iteration)

        self._write_summary(summary)
