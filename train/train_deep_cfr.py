"""
DeepCFRTrainer — entrenamiento con redes neuronales para CFR

BUGS CORREGIDOS + OPTIMIZACIONES para Ryzen 7 5800X + RTX 3060 12GB
─────────────────────────────────────────────────────────────────────────────
BUG-7  Encoding de estado con 4 features SIN información de cartas.
       La red neuronal no podía distinguir AA de 72o.
       FIX: añadido card_bucket normalizado como 5ª feature.

BUG-8  Entrenamiento muestra a muestra (batch_size=1) — GPU infrautilizada.
       FIX: DataLoader con batch_size=512, pin_memory=True, num_workers=4.

BUG-9  Sin mixed precision → RTX 3060 trabajando a la mitad de su potencia.
       FIX: torch.cuda.amp (autocast + GradScaler).

BUG-10 Sin gradient clipping → posibles explosions de gradiente en regret net.
       FIX: clip_grad_norm_ con max_norm=1.0.

BUG-11 Sin LR scheduling → convergencia lenta y oscilaciones tarde en training.
       FIX: CosineAnnealingLR.

BUG-12 Inconsistencia entre encode_state() en encoding.py (pot/200) y el
       _encode_state() inline (pot raw) → features distintas según desde donde
       se llamara.  FIX: un único método canónico.
─────────────────────────────────────────────────────────────────────────────
OPTIMIZACIONES HARDWARE (Ryzen 7 5800X + RTX 3060 12 GB):
  - torch.compile (PyTorch 2.x) para redes ~15-25% más rápidas
  - num_workers=4 en DataLoader (8 cores → 4 workers + 4 para Python)
  - pin_memory=True para transfers CPU→GPU sin copia intermedia
  - AMP float16 en GPU para doblar el throughput de la RTX 3060
  - Batch size 512 → alta ocupación de CUDA cores sin OOM en 12 GB VRAM
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np

from models.regret_net import RegretNet
from models.policy_net import PolicyNet
from data.buffers import ReservoirBuffer
from data.dataset import AdvantageDataset, PolicyDataset
from utils.information_set import InformationSetBuilder
from utils.logging import log_metric

NUM_ACTIONS = 3
ACTION_LIST = ["fold", "call", "raise"]

# ─────────────────────────────────────────────────────────────────────────────
# Feature dimension.  Debe coincidir con lo que devuelve _encode_state().
# 5 features: pot, player, board_len, stack_ratio, card_bucket
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIM = 5


class DeepCFRTrainer:

    def __init__(self, env):
        self.env = env
        self.iss = InformationSetBuilder()
        self.actions = ACTION_LIST

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DeepCFR] Usando dispositivo: {self.device}")

        # Redes con dimensiones correctas
        self.regret_net = RegretNet(
            input_dim=FEATURE_DIM, output_dim=NUM_ACTIONS
        ).to(self.device)

        self.policy_net = PolicyNet(
            input_dim=FEATURE_DIM, output_dim=NUM_ACTIONS
        ).to(self.device)

        # OPTIMIZACIÓN: compilar modelos con torch.compile (PyTorch >= 2.0)
        if hasattr(torch, "compile"):
            try:
                self.regret_net = torch.compile(self.regret_net)
                self.policy_net = torch.compile(self.policy_net)
                print("[DeepCFR] torch.compile activado")
            except Exception:
                pass  # Fallback silencioso si el backend no soporta compile

        self.advantage_buffer = ReservoirBuffer(max_size=100_000)
        self.policy_buffer = ReservoirBuffer(max_size=100_000)

        self.regret_opt = optim.Adam(self.regret_net.parameters(), lr=1e-3,
                                     weight_decay=1e-5)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=1e-3,
                                     weight_decay=1e-5)

        # LR schedulers — cosine annealing para convergencia suave
        self.regret_sched = optim.lr_scheduler.CosineAnnealingLR(
            self.regret_opt, T_max=20, eta_min=1e-5
        )
        self.policy_sched = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_opt, T_max=20, eta_min=1e-5
        )

        # Scaler para AMP (solo activo en CUDA)
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

    # -------------------------------------------------------------------------
    # SELF-PLAY
    # -------------------------------------------------------------------------
    def self_play(self, episodes: int = 100) -> None:
        self.regret_net.eval()

        for _ in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                x = self._encode_state(state).to(self.device).unsqueeze(0)

                with torch.no_grad():
                    with autocast(enabled=(self.device.type == "cuda")):
                        regrets = self.regret_net(x).squeeze(0).cpu().float().numpy()

                strategy = self._regret_to_strategy(regrets)
                action_str = self._sample_action(strategy)

                state, reward, done, _ = self.env.step(action_str)

                x_cpu = x.squeeze(0).cpu().float()
                self.policy_buffer.add((x_cpu, strategy.copy()))

                for i, adv in enumerate(regrets):
                    self.advantage_buffer.add((x_cpu, i, float(adv)))

    # -------------------------------------------------------------------------
    # FEATURE ENCODING  (canónico — única versión en todo el proyecto)
    #
    # BUG-7 FIX: incluida bucket de cartas (5ª feature) normalizada a [0,1].
    # Sin esta feature, la red no puede diferenciar manos fuertes de débiles.
    # -------------------------------------------------------------------------
    def _encode_state(self, state: dict) -> torch.Tensor:
        starting_stack = getattr(self.env, "starting_stack", 100)
        max_pot = 2 * starting_stack

        pot = float(state.get("pot", 0)) / max_pot
        player = float(state.get("current_player", 0))
        board_len = float(len(state.get("board", []))) / 5.0

        stacks = state.get("stacks", [starting_stack] * 2)
        total = sum(stacks) + state.get("pot", 0) + 1e-8
        stack_ratio = float(stacks[state.get("current_player", 0)]) / total

        # BUG-7 FIX: card bucket desde la abstracción de cartas
        player_idx = state.get("current_player", 0)
        hand = state.get("hands", [[], []])[player_idx]
        board = state.get("board", [])
        num_buckets = self.iss.card_abs.num_buckets  # típicamente 10
        if hand:
            bucket = self.iss.card_abs.bucket_hand(hand, board)
            card_feature = float(bucket) / max(num_buckets - 1, 1)
        else:
            card_feature = 0.5   # prior neutro si no hay cartas todavía

        return torch.tensor(
            [pot, player, board_len, stack_ratio, card_feature],
            dtype=torch.float32
        )

    # -------------------------------------------------------------------------
    # STRATEGY
    # -------------------------------------------------------------------------
    def _regret_to_strategy(self, regrets: np.ndarray) -> np.ndarray:
        pos = np.maximum(regrets, 0.0)
        total = pos.sum()
        if total > 1e-8:
            return pos / total
        return np.ones(len(self.actions)) / len(self.actions)

    def _sample_action(self, strategy: np.ndarray) -> str:
        legal = self.env.get_legal_actions()
        legal_idx = [self.actions.index(a) for a in legal if a in self.actions]
        sub = strategy[legal_idx]
        sub = sub / (sub.sum() + 1e-8)
        chosen = np.random.choice(legal_idx, p=sub)
        return self.actions[chosen]

    # -------------------------------------------------------------------------
    # REGRET NET TRAINING  (BUG-8 FIX: batch training con DataLoader)
    # -------------------------------------------------------------------------
    def train_regret_net(self, epochs: int = 5, batch_size: int = 512) -> float:
        if len(self.advantage_buffer) < batch_size:
            return 0.0

        self.regret_net.train()
        dataset = AdvantageDataset(self.advantage_buffer)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,          # Ryzen 5800X: 4 workers CPU paralelo
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        total_loss = 0.0
        steps = 0

        for _ in range(epochs):
            for x_batch, a_batch, adv_batch in loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                a_batch = a_batch.to(self.device, non_blocking=True)
                adv_batch = adv_batch.to(self.device, non_blocking=True)

                with autocast(enabled=(self.device.type == "cuda")):
                    preds = self.regret_net(x_batch)           # [B, num_actions]
                    # Loss enmascarada: solo en la acción observada
                    pred_for_action = preds.gather(
                        1, a_batch.unsqueeze(1)
                    ).squeeze(1)
                    loss = ((pred_for_action - adv_batch) ** 2).mean()

                self.regret_opt.zero_grad()
                self.scaler.scale(loss).backward()
                # BUG-10 FIX: gradient clipping
                self.scaler.unscale_(self.regret_opt)
                nn.utils.clip_grad_norm_(self.regret_net.parameters(), 1.0)
                self.scaler.step(self.regret_opt)
                self.scaler.update()

                total_loss += loss.item()
                steps += 1

        self.regret_sched.step()
        return total_loss / max(steps, 1)

    # -------------------------------------------------------------------------
    # POLICY NET TRAINING  (BUG-8 FIX: batch training con DataLoader)
    # -------------------------------------------------------------------------
    def train_policy_net(self, epochs: int = 5, batch_size: int = 512) -> float:
        if len(self.policy_buffer) < batch_size:
            return 0.0

        self.policy_net.train()
        dataset = PolicyDataset(self.policy_buffer)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        loss_fn = nn.MSELoss()
        total_loss = 0.0
        steps = 0

        for _ in range(epochs):
            for x_batch, strat_batch in loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                strat_batch = strat_batch.to(self.device, non_blocking=True)

                with autocast(enabled=(self.device.type == "cuda")):
                    pred = self.policy_net(x_batch)
                    loss = loss_fn(pred, strat_batch)

                self.policy_opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.policy_opt)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.scaler.step(self.policy_opt)
                self.scaler.update()

                total_loss += loss.item()
                steps += 1

        self.policy_sched.step()
        return total_loss / max(steps, 1)

    # -------------------------------------------------------------------------
    # EV EVALUATION
    # -------------------------------------------------------------------------
    def evaluate_ev(self, num_hands: int = 1000) -> tuple:
        self.regret_net.eval()
        total_reward = 0.0
        wins = 0

        for _ in range(num_hands):
            state = self.env.reset()
            done = False

            while not done:
                x = self._encode_state(state).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    with autocast(enabled=(self.device.type == "cuda")):
                        regrets = self.regret_net(x).squeeze(0).cpu().float().numpy()
                strategy = self._regret_to_strategy(regrets)
                action = self._sample_action(strategy)
                state, reward, done, _ = self.env.step(action)

            total_reward += reward
            if reward > 0:
                wins += 1

        ev = total_reward / num_hands
        winrate = wins / num_hands
        return ev, winrate

    # -------------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -------------------------------------------------------------------------
    def train(self, iterations: int = 50, episodes_per_iter: int = 200,
              eval_hands: int = 1000, batch_size: int = 512) -> None:

        print(f"[DeepCFR] Iniciando entrenamiento: {iterations} iters, "
              f"{episodes_per_iter} eps/iter, batch={batch_size}")

        for i in range(iterations):
            self.self_play(episodes=episodes_per_iter)

            r_loss = self.train_regret_net(batch_size=batch_size)
            p_loss = self.train_policy_net(batch_size=batch_size)
            ev, winrate = self.evaluate_ev(num_hands=eval_hands)

            log_metric("regret_loss", r_loss, step=i)
            log_metric("policy_loss", p_loss, step=i)
            log_metric("ev", ev, step=i)
            log_metric("winrate", winrate, step=i)

            r_lr = self.regret_opt.param_groups[0]["lr"]
            p_lr = self.policy_opt.param_groups[0]["lr"]
            print(
                f"[DeepCFR] Iter {i:03d} | EV={ev:+.4f} | WR={winrate:.4f} "
                f"| R-loss={r_loss:.5f} | P-loss={p_loss:.5f} "
                f"| r_lr={r_lr:.2e} | p_lr={p_lr:.2e} "
                f"| adv_buf={len(self.advantage_buffer):,} "
                f"| pol_buf={len(self.policy_buffer):,}"
            )
