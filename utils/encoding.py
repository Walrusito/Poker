"""
encode_state — vector de features canónico para el information set.

BUG-12 FIX: existían dos implementaciones diferentes del encoding:
  1. encode_state() en este fichero: usaba pot/200 (normalizado)
  2. _encode_state() en train_deep_cfr.py: usaba pot raw sin normalizar
  Las dos divergían y podían usarse en distintas partes del código → redes
  recibían features con escalas diferentes sin ser conscientes de ello.

  FIX: una única función canónica aquí. train_deep_cfr.py llama a su propio
  método _encode_state() que sigue exactamente el mismo contrato.

Features (5 valores, todos ∈ [0, 1]):
  0  pot / (2 * starting_stack)        — pot normalizado
  1  current_player                    — 0 o 1
  2  board_len / 5                     — 0 (preflop) → 1 (river)
  3  stack_ratio del jugador activo    — su stack / chips totales
  4  card_bucket / (num_buckets - 1)   — fuerza de mano normalizada [0,1]
"""

import numpy as np
from typing import Dict, Any, Optional


def encode_state(state: Dict[str, Any],
                 card_abstraction=None,
                 starting_stack: int = 100,
                 num_buckets: int = 10) -> np.ndarray:
    """
    Retorna un np.ndarray de shape (5,) con dtype float32.

    Parámetros:
        state            — dict del estado de la env
        card_abstraction — instancia de CardAbstraction (opcional)
                           si es None, la 5ª feature será 0.5 (prior neutro)
        starting_stack   — stack inicial para normalizar el pot
        num_buckets      — número de buckets de la card abstraction
    """
    max_pot = max(2 * starting_stack, 1)
    pot = float(state.get("pot", 0)) / max_pot
    player = float(state.get("current_player", 0))
    board_len = float(len(state.get("board", []))) / 5.0

    stacks = state.get("stacks", [starting_stack] * 2)
    total = sum(stacks) + state.get("pot", 0) + 1e-8
    stack_ratio = float(stacks[state.get("current_player", 0)]) / total

    if card_abstraction is not None:
        p = state.get("current_player", 0)
        hand = state.get("hands", [[], []])[p]
        board = state.get("board", [])
        if hand:
            bucket = card_abstraction.bucket_hand(hand, board)
            card_feat = float(bucket) / max(num_buckets - 1, 1)
        else:
            card_feat = 0.5
    else:
        card_feat = 0.5

    return np.array([pot, player, board_len, stack_ratio, card_feat],
                    dtype=np.float32)
