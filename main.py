"""
main.py — punto de entrada del entrenamiento Deep CFR

Defaults optimizados para Ryzen 7 5800X + RTX 3060 12 GB:
  --iterations 50       : suficientes para convergencia básica
  --episodes  200       : más self-play por iter para llenar los buffers
  --eval-hands 500      : balance velocidad / precisión en evaluación
  --batch-size 512      : alta ocupación CUDA en RTX 3060 sin OOM
"""

import argparse
import torch

from env.poker_env import PokerEnv
from train.train_deep_cfr import DeepCFRTrainer
from utils.logging import start_experiment, end_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Poker CFR AI — Deep CFR Training")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--episodes", type=int, default=200,
                   help="Self-play episodes por iteración")
    p.add_argument("--eval-hands", type=int, default=500,
                   help="Manos para evaluar EV por iteración")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size para entrenar las redes (GPU)")
    p.add_argument("--experiment", type=str, default="poker_cfr_optimized")
    p.add_argument("--starting-stack", type=int, default=100)
    p.add_argument("--seed", type=int, default=None,
                   help="Semilla para reproducibilidad")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Información de hardware
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[main] GPU detectada: {gpu} ({vram:.1f} GB VRAM)")
    else:
        print("[main] CUDA no disponible — entrenando en CPU")

    env = PokerEnv(num_players=2, starting_stack=args.starting_stack)
    trainer = DeepCFRTrainer(env)

    start_experiment(args.experiment)

    trainer.train(
        iterations=args.iterations,
        episodes_per_iter=args.episodes,
        eval_hands=args.eval_hands,
        batch_size=args.batch_size,
    )

    end_experiment()
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
