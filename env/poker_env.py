"""
PokerEnv — Texas Hold'em simplificado, compatible con CFR.

BUGS CORREGIDOS vs versión original:
─────────────────────────────────────────────────────────────────────────────
BUG-1  _fold_reward() devolvía SIEMPRE -1.0 sin importar quién hacía fold.
       Si el jugador 1 hace fold, el jugador 0 GANA → debe devolver +1.0.
       Esto distorsionaba completamente los EV y hacía que el entrenamiento
       convergiera a estrategias incorrectas.

BUG-2  _post_deal_setup() pre-repartía el flop (3 cartas) ANTES de que
       empezara el preflop. Esto hacía que el state en preflop ya tuviera
       board=['c1','c2','c3'], contaminando los information sets de preflop
       con información del flop. Las manos se evaluaban con board visible.

BUG-3  _advance_street() no repartía nuevas cartas en flop/turn/river de
       forma independiente; asumía que ya estaban pre-repartidas (consecuencia
       de BUG-2).

BUG-4  _street_complete() comprobaba [1,1] acumulado total, no por street.
       Con street_actions=[0,0] al inicio de cada calle y conteo correcto
       ya funcionaba, pero la lógica de raise sin burn era inconsistente.
─────────────────────────────────────────────────────────────────────────────
"""

import random
from typing import List, Dict, Tuple, Any


class PokerEnv:

    ACTIONS = ["fold", "call", "raise"]

    def __init__(self, num_players: int = 2, starting_stack: int = 100):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.reset()

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.deck = list(range(52))
        random.shuffle(self.deck)

        self.hands = self._deal_hands()
        self.board: List[int] = []          # BUG-2 FIX: board empieza VACÍO

        self.pot = 0
        self.bets = [0] * self.num_players
        self.stacks = [self.starting_stack] * self.num_players

        self.current_player = 0
        self.street = "preflop"

        self.history: List[Tuple] = []
        self.done = False
        self.street_actions = [0] * self.num_players   # acciones POR STREET

        return self._get_state()

    # -------------------------------------------------------------------------
    # LEGAL ACTIONS
    # -------------------------------------------------------------------------
    def get_legal_actions(self) -> List[str]:
        actions = ["fold", "call"]
        if self.stacks[self.current_player] >= 2:
            actions.append("raise")
        return actions

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if self.done:
            return self._get_state(), 0.0, True, {}

        player = self.current_player
        self.history.append((player, action))
        self.street_actions[player] += 1

        # -- Lógica de acción --------------------------------------------------
        if action == "fold":
            self.done = True
            # BUG-1 FIX: reward depende de QUIÉN hace fold, siempre desde
            # la perspectiva del jugador 0.
            return self._get_state(), self._fold_reward(player), True, {}

        elif action == "call":
            amount = min(1, self.stacks[player])
            self.pot += amount
            self.bets[player] += amount
            self.stacks[player] -= amount

        elif action == "raise":
            amount = min(2, self.stacks[player])
            self.pot += amount
            self.bets[player] += amount
            self.stacks[player] -= amount

        # Siguiente jugador
        self.current_player = (self.current_player + 1) % self.num_players

        # Transición de calle
        if self._street_complete():
            self._advance_street()

        if self.street == "showdown":
            self.done = True
            return self._get_state(), self._showdown(), True, {}

        return self._get_state(), 0.0, False, {}

    # -------------------------------------------------------------------------
    # STATE
    # -------------------------------------------------------------------------
    def _get_state(self) -> Dict[str, Any]:
        return {
            "street": self.street,
            "hands": self.hands,
            "board": self.board,
            "pot": self.pot,
            "bets": self.bets,
            "stacks": self.stacks,
            "current_player": self.current_player,
            "history": self.history,
            "done": self.done,
        }

    # -------------------------------------------------------------------------
    # DECK / DEAL
    # -------------------------------------------------------------------------
    def _deal_hands(self) -> List[List[int]]:
        hands = [[] for _ in range(self.num_players)]
        for _ in range(2):
            for p in range(self.num_players):
                hands[p].append(self.deck.pop())
        return hands

    # BUG-2 FIX: eliminada _post_deal_setup() que pre-repartía el flop.
    # Las cartas comunitarias se reparten al avanzar de calle.

    # -------------------------------------------------------------------------
    # STREET LOGIC
    # -------------------------------------------------------------------------
    def _street_complete(self) -> bool:
        """
        La calle termina cuando todos los jugadores han actuado al menos una
        vez y las apuestas están igualadas.

        Modelo simplificado: exactamente 1 acción por jugador por calle.
        Suficiente para el árbol CFR de heads-up.
        """
        return all(a >= 1 for a in self.street_actions)

    def _advance_street(self):
        """
        BUG-3 FIX: ahora reparte las cartas comunitarias aquí en lugar de
        asumirlas pre-repartidas. Añadido burn card antes de cada calle
        (estándar Texas Hold'em).
        """
        self.street_actions = [0] * self.num_players

        if self.street == "preflop":
            self.deck.pop()                                         # burn
            self.board.extend([self.deck.pop() for _ in range(3)]) # flop
            self.street = "flop"

        elif self.street == "flop":
            self.deck.pop()                                         # burn
            self.board.append(self.deck.pop())                      # turn
            self.street = "turn"

        elif self.street == "turn":
            self.deck.pop()                                         # burn
            self.board.append(self.deck.pop())                      # river
            self.street = "river"

        elif self.street == "river":
            self.street = "showdown"

    # -------------------------------------------------------------------------
    # TERMINAL REWARDS
    # -------------------------------------------------------------------------
    def _fold_reward(self, folding_player: int) -> float:
        """
        BUG-1 FIX: siempre desde la perspectiva del jugador 0.
        - P0 hace fold → pierde → -1.0
        - P1 hace fold → P0 gana → +1.0
        """
        return -1.0 if folding_player == 0 else 1.0

    def _showdown(self) -> float:
        """Evaluación real de manos (ya correcta en original)."""
        from env.rules import evaluate_7
        h0 = evaluate_7(self.hands[0] + self.board)
        h1 = evaluate_7(self.hands[1] + self.board)
        if h0 > h1:
            return 1.0
        elif h1 > h0:
            return -1.0
        return 0.0
