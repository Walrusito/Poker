"""
HandEquity — Estimador de equidad por Monte Carlo

Optimizaciones aplicadas:
  - Vectorización con NumPy para generación de deals y evaluación
  - Soporte multiway (num_players > 2)
  - Cache por (hand, board, num_players) — evita recalcular mismas manos
  - Simulaciones adaptativas por street (Step 7):
      preflop: todas las sims (LUT debería cubrir la mayoría)
      flop/turn: sims reducidas proporcionalmente
      river: determinístico cuando no hay cartas por repartir
  - evaluate_7_cached con LRU para reutilizar evaluaciones de 7 cartas
"""

import numpy as np
from typing import List, Optional

from env.rules import evaluate_7, evaluate_7_cached


class HandEquity:

    def __init__(self, simulations: int = 200, seed: Optional[int] = None,
                 use_torch_backend: bool = False, torch_device: str = "cpu"):
        self.simulations = simulations
        self._rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def estimate(self, hand: List[int], board: Optional[List[int]] = None,
                 num_players: int = 2) -> float:
        if board is None:
            board = []

        key = (tuple(sorted(hand)), tuple(board), num_players)
        if key in self._cache:
            return self._cache[key]

        known = set(hand) | set(board)
        available = np.array([c for c in range(52) if c not in known], dtype=np.int32)

        cards_needed = 5 - len(board)
        n_opponents = num_players - 1

        # River with complete board: deterministic evaluation, no MC needed
        if cards_needed == 0 and n_opponents == 1:
            equity = self._exact_river_equity(hand, board, n_opponents, available)
            self._cache[key] = equity
            return equity

        # Adaptive simulation count by street (Step 7)
        n_sims = self._adaptive_sims(len(board))
        cards_per_sim = 2 * n_opponents + cards_needed

        if len(available) < cards_per_sim:
            self._cache[key] = 0.5
            return 0.5

        # Vectorised deal generation: shuffle indices, slice cards
        indices = np.argsort(
            self._rng.random((n_sims, len(available))), axis=1
        )
        deals = available[indices[:, :cards_per_sim]]

        board_arr = np.array(board, dtype=np.int32)
        hand_arr = np.array(hand, dtype=np.int32)

        wins = 0
        ties = 0

        for i in range(n_sims):
            deal = deals[i]
            board_completion = list(board_arr) + list(deal[2 * n_opponents:])
            hero_cards = tuple(sorted(list(hand_arr) + board_completion))
            hero_score = evaluate_7_cached(hero_cards)

            hero_wins_all = True
            hero_ties_all = True
            for opp_idx in range(n_opponents):
                opp_hand = deal[2 * opp_idx: 2 * opp_idx + 2]
                opp_cards = tuple(sorted(list(opp_hand) + board_completion))
                opp_score = evaluate_7_cached(opp_cards)

                if hero_score <= opp_score:
                    hero_wins_all = False
                if hero_score != opp_score:
                    hero_ties_all = False

            if hero_wins_all:
                wins += 1
            elif hero_ties_all:
                ties += 1

        equity = (wins + 0.5 * ties) / n_sims
        self._cache[key] = equity
        return equity

    def _adaptive_sims(self, board_len: int) -> int:
        """Reduce simulations on earlier streets where precision is less critical."""
        if board_len == 5:
            return max(50, self.simulations // 4)
        if board_len == 4:
            return max(80, self.simulations // 2)
        if board_len == 3:
            return max(100, (self.simulations * 3) // 4)
        return self.simulations

    def _exact_river_equity(self, hand, board, n_opponents, available) -> float:
        """Exact heads-up river equity: enumerate all possible opponent hands."""
        hero_cards = tuple(sorted(hand + board))
        hero_score = evaluate_7_cached(hero_cards)

        wins = 0
        ties = 0
        total = 0

        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                opp_hand = [available[i], available[j]]
                opp_cards = tuple(sorted(opp_hand + board))
                opp_score = evaluate_7_cached(opp_cards)

                if hero_score > opp_score:
                    wins += 1
                elif hero_score == opp_score:
                    ties += 1
                total += 1

        if total == 0:
            return 0.5
        return (wins + 0.5 * ties) / total

    def clear_cache(self) -> None:
        self._cache.clear()
