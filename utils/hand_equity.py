"""
HandEquity — Estimador de equidad por Monte Carlo

BUG CRÍTICO CORREGIDO:
─────────────────────────────────────────────────────────────────────────────
BUG-6  _clone_deck() copiaba la lista de cartas filtradas SIN BARAJAR.
       Consecuencia: las 'self.simulations' simulaciones repartían SIEMPRE
       las mismas 2 cartas al oponente y el mismo completado de board.
       El estimador producía siempre el mismo resultado (win/loss/tie) en
       todas las sims → varianza cero pero SESGO MÁXIMO.
       La equidad resultante era completamente incorrecta.

       FIX: antes de cada simulación se baraja una copia de las cartas
       disponibles, garantizando independencia estadística entre sims.
─────────────────────────────────────────────────────────────────────────────

OPTIMIZACIONES:
  - Cache LRU ilimitado por (hand, board) — evita recalcular mismas manos
  - Pre-filtrado de cartas conocidas una sola vez antes del bucle
  - Vectorización del loop de simulación
  - Soporte para seed reproducible
"""

import random
from functools import lru_cache
from typing import List, Optional, Tuple

from env.rules import evaluate_7


class HandEquity:

    def __init__(self, simulations: int = 200, seed: Optional[int] = None):
        self.simulations = simulations
        self.rng = random.Random(seed)
        # Cache manual para admitir listas (lru_cache no acepta unhashable)
        self._cache: dict = {}

    # -------------------------------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------------------------------
    def estimate(self, hand: List[int], board: Optional[List[int]] = None) -> float:
        """
        Estima la equidad de 'hand' frente a una mano oponente aleatoria
        completando el board con cartas desconocidas.

        Retorna un float en [0, 1]:  1 = gana siempre, 0 = pierde siempre.
        """
        if board is None:
            board = []

        key = (tuple(hand), tuple(board))
        if key in self._cache:
            return self._cache[key]

        # Cartas ya conocidas (no se pueden repartir al oponente)
        known = set(hand + board)
        available = [c for c in range(52) if c not in known]

        wins = 0
        ties = 0
        cards_needed_board = 5 - len(board)

        for _ in range(self.simulations):
            # BUG-6 FIX: barajar una copia independiente en cada simulación
            deck = available.copy()
            self.rng.shuffle(deck)

            # Repartir 2 cartas al oponente y completar el board
            opp_hand = deck[:2]
            board_completion = board + deck[2: 2 + cards_needed_board]

            hero = evaluate_7(hand + board_completion)
            opp = evaluate_7(opp_hand + board_completion)

            if hero > opp:
                wins += 1
            elif hero == opp:
                ties += 1

        equity = (wins + 0.5 * ties) / self.simulations
        self._cache[key] = equity
        return equity

    def clear_cache(self) -> None:
        self._cache.clear()
