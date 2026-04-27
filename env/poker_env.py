import random
from typing import Any, Dict, List, Tuple

from env.rules import evaluate_7
from utils.action_abstraction import DEFAULT_ACTIONS
from utils.bet_sizing_abstraction import BetSizingAbstraction
from utils.position import Position


class PokerEnv:
    DEFAULT_ACTIONS_LIST = list(DEFAULT_ACTIONS)

    __slots__ = (
        "num_players", "starting_stack", "small_blind", "big_blind",
        "reward_unit", "seed", "rng", "bet_sizing", "_actions", "position",
        "hand_index", "reference_player", "deck", "hands", "board",
        "pot", "contributions", "bets", "stacks", "active", "street",
        "history", "done", "pending_players", "current_player",
        "last_raise_size", "last_aggressor", "last_action",
        "street_actions", "last_terminal_utilities", "last_winners",
    )

    @property
    def ACTIONS(self):
        return self._actions

    @ACTIONS.setter
    def ACTIONS(self, value):
        self._actions = value

    def __init__(
        self,
        num_players: int = 2,
        starting_stack: int = 10000,
        small_blind: int = 50,
        big_blind: int = 100,
        reward_unit: str = "bb",
        street_bet_multipliers=None,
        street_raise_multipliers=None,
        seed=None,
    ):
        if not 2 <= num_players <= 9:
            raise ValueError("num_players must be between 2 and 9")

        self.num_players = num_players
        self.starting_stack = float(starting_stack)
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.reward_unit = reward_unit
        self.seed = seed
        self.rng = random.Random(seed)
        self.bet_sizing = BetSizingAbstraction(
            street_bet_multipliers=street_bet_multipliers,
            street_raise_multipliers=street_raise_multipliers,
        )
        self.ACTIONS = list(self.bet_sizing.actions)
        self.position = Position(num_players)
        self.hand_index = 0
        self.reference_player = 0
        self.reset()

    def reset(self) -> Dict[str, Any]:
        if self.hand_index > 0:
            self.position.switch_button()
        self.hand_index += 1

        self.deck = self._create_deck()
        self.rng.shuffle(self.deck)

        self.hands = self._deal_hands()
        self.board: List[int] = []

        self.pot = 0.0
        self.contributions = [0.0 for _ in range(self.num_players)]
        self.bets = [0.0 for _ in range(self.num_players)]
        self.stacks = [self.starting_stack for _ in range(self.num_players)]
        self.active = [True for _ in range(self.num_players)]

        self.street = "preflop"
        self.history = []
        self.done = False
        self.pending_players: List[int] = []
        self.current_player = 0
        self.last_raise_size = self.big_blind
        self.last_aggressor = None
        self.last_action = None
        self.street_actions = 0
        self.last_terminal_utilities = [0.0 for _ in range(self.num_players)]
        self.last_winners: List[int] = []

        self._post_blinds()
        self._initialize_preflop_round()
        return self._get_state()

    def get_legal_actions(self) -> List[str]:
        if self.done:
            return []

        player = self.current_player
        if not self.active[player] or self.stacks[player] <= 0:
            return []

        to_call = max(self.bets) - self.bets[player]
        return self.bet_sizing.get_actions(
            pot=self.pot,
            stack=self.stacks[player],
            min_raise=self.last_raise_size,
            to_call=to_call,
            street=self.street,
            big_blind=self.big_blind,
        )

    def clone(self):
        new = object.__new__(PokerEnv)
        new.num_players = self.num_players
        new.starting_stack = self.starting_stack
        new.small_blind = self.small_blind
        new.big_blind = self.big_blind
        new.reward_unit = self.reward_unit
        new.seed = self.seed
        new.rng = self.rng
        new.bet_sizing = self.bet_sizing
        new._actions = self._actions
        new.position = self.position.clone()
        new.hand_index = self.hand_index
        new.reference_player = self.reference_player
        new.deck = self.deck[:]
        new.hands = [h[:] for h in self.hands]
        new.board = self.board[:]
        new.pot = self.pot
        new.contributions = self.contributions[:]
        new.bets = self.bets[:]
        new.stacks = self.stacks[:]
        new.active = self.active[:]
        new.street = self.street
        new.history = self.history[:]
        new.done = self.done
        new.pending_players = self.pending_players[:]
        new.current_player = self.current_player
        new.last_raise_size = self.last_raise_size
        new.last_aggressor = self.last_aggressor
        new.last_action = self.last_action
        new.street_actions = self.street_actions
        new.last_terminal_utilities = self.last_terminal_utilities[:]
        new.last_winners = self.last_winners[:]
        return new

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if self.done:
            return self._get_state(), 0.0, True, {}

        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            raise ValueError(f"Illegal action '{action}' for player {self.current_player}: {legal_actions}")

        player = self.current_player
        to_call = max(self.bets) - self.bets[player]

        self.history.append((player, action))
        self.last_action = action
        self.street_actions += 1

        if action == "fold":
            self.active[player] = False
            self._remove_from_pending(player)
            if self._active_player_count() == 1:
                return self._finish_single_winner()

        elif action == "check":
            self._remove_from_pending(player)

        elif action == "call":
            amount = self.bet_sizing.to_amount(action, self.pot, self.stacks[player], self.last_raise_size, to_call)
            self._commit_chips(player, amount)
            self._remove_from_pending(player)

        else:
            amount = self._bet_amount(player, action, to_call)
            raise_size = max(0.0, amount - to_call)
            self._commit_chips(player, amount)
            if raise_size > 0.0:
                self.last_raise_size = max(self.big_blind, raise_size)
                self.last_aggressor = player
                self._reset_pending_after_aggression(player)
            else:
                self._remove_from_pending(player)

        if self._street_complete():
            return self._advance_after_round()

        self.current_player = self.pending_players[0]
        return self._get_state(), 0.0, False, {}

    def get_terminal_utilities(self) -> List[float]:
        return self.last_terminal_utilities

    def _get_state(self) -> Dict[str, Any]:
        return {
            "street": self.street,
            "hands": self.hands,
            "board": self.board,
            "pot": self.pot,
            "bets": self.bets,
            "contributions": self.contributions,
            "stacks": self.stacks,
            "active": self.active,
            "current_player": self.current_player,
            "pending_players": self.pending_players,
            "button": self.position.button,
            "starting_stack": self.starting_stack,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "last_raise_size": self.last_raise_size,
            "last_aggressor": self.last_aggressor,
            "last_action": self.last_action,
            "street_actions": self.street_actions,
            "num_players": self.num_players,
            "history": self.history,
            "done": self.done,
        }

    def _create_deck(self) -> List[int]:
        return list(range(52))

    def _deal_hands(self) -> List[List[int]]:
        hands = [[] for _ in range(self.num_players)]
        for _ in range(2):
            for player in range(self.num_players):
                hands[player].append(self.deck.pop())
        return hands

    def _post_blinds(self):
        sb_player, bb_player = self.position.blind_positions()
        self._commit_chips(sb_player, min(self.small_blind, self.stacks[sb_player]))
        self._commit_chips(bb_player, min(self.big_blind, self.stacks[bb_player]))
        self.last_raise_size = self.big_blind
        self.last_aggressor = bb_player
        self.last_action = "blind"

    def _initialize_preflop_round(self):
        first = self.position.first_to_act_preflop(self.active)
        self.pending_players = self._ordered_active_players(first)
        self.current_player = self.pending_players[0]

    def _ordered_active_players(self, start_player: int) -> List[int]:
        order = []
        for offset in range(self.num_players):
            seat = (start_player + offset) % self.num_players
            if self.active[seat] and self.stacks[seat] > 0:
                order.append(seat)
        return order

    def _remove_from_pending(self, player: int):
        self.pending_players = [seat for seat in self.pending_players if seat != player]

    def _reset_pending_after_aggression(self, aggressor: int):
        start = (aggressor + 1) % self.num_players
        self.pending_players = self._ordered_active_players(start)
        self._remove_from_pending(aggressor)

    def _bet_amount(self, player: int, action: str, to_call: float) -> float:
        return self.bet_sizing.to_amount(
            action=action,
            pot=self.pot,
            stack=self.stacks[player],
            min_raise=self.last_raise_size,
            to_call=to_call,
            street=self.street,
            big_blind=self.big_blind,
        )

    def _commit_chips(self, player: int, amount: float):
        self.pot += amount
        self.bets[player] += amount
        self.contributions[player] += amount
        self.stacks[player] -= amount

    def _street_complete(self) -> bool:
        return len(self.pending_players) == 0

    def _advance_after_round(self) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if self._active_player_count() == 1:
            return self._finish_single_winner()

        if self._active_players_with_chips() <= 1:
            self._runout_to_showdown()
            return self._finish_showdown()

        if self.street == "river":
            self.street = "showdown"
            return self._finish_showdown()

        self._advance_street()
        if self.done:
            return self._finish_showdown()

        return self._get_state(), 0.0, False, {}

    def _advance_street(self):
        self.bets = [0.0 for _ in range(self.num_players)]
        self.last_raise_size = self.big_blind
        self.last_aggressor = None
        self.last_action = None
        self.street_actions = 0

        if self.street == "preflop":
            self._burn_card()
            self.board.extend([self.deck.pop() for _ in range(3)])
            self.street = "flop"
        elif self.street == "flop":
            self._burn_card()
            self.board.append(self.deck.pop())
            self.street = "turn"
        elif self.street == "turn":
            self._burn_card()
            self.board.append(self.deck.pop())
            self.street = "river"
        else:
            self.street = "showdown"
            self.done = True
            return

        first = self.position.first_to_act_postflop(self.active)
        self.pending_players = self._ordered_active_players(first)

        if len(self.pending_players) <= 1:
            self._runout_to_showdown()
            self.done = True
            return

        self.current_player = self.pending_players[0]

    def _runout_to_showdown(self):
        while len(self.board) < 5:
            self._burn_card()
            self.board.append(self.deck.pop())
        self.street = "showdown"
        self.done = True

    def _burn_card(self):
        if self.deck:
            self.deck.pop()

    def _finish_single_winner(self):
        winner = next(player for player, is_active in enumerate(self.active) if is_active)
        payouts = [0.0 for _ in range(self.num_players)]
        payouts[winner] = self.pot
        return self._finish_hand(payouts, winners=[winner])

    def _finish_showdown(self):
        payouts = self._compute_showdown_payouts()
        winners = [player for player, payout in enumerate(payouts) if payout > 0 and self.active[player]]
        return self._finish_hand(payouts, winners=winners)

    def _compute_showdown_payouts(self):
        scores = {
            player: evaluate_7(self.hands[player] + self.board)
            for player, is_active in enumerate(self.active)
            if is_active
        }
        remaining = self.contributions.copy()
        payouts = [0.0 for _ in range(self.num_players)]

        while any(amount > 0 for amount in remaining):
            layer = min(amount for amount in remaining if amount > 0)
            participants = [player for player, amount in enumerate(remaining) if amount > 0]
            layer_pot = layer * len(participants)

            for player in participants:
                remaining[player] -= layer

            contenders = [player for player in participants if self.active[player]]
            best_score = max(scores[player] for player in contenders)
            layer_winners = [player for player in contenders if scores[player] == best_score]
            share = layer_pot / len(layer_winners)

            for winner in layer_winners:
                payouts[winner] += share

        return payouts

    def _finish_hand(self, payouts, winners):
        self.done = True
        self.last_winners = winners
        self.last_terminal_utilities = []
        total_starting_chips = self.starting_stack * self.num_players

        for player in range(self.num_players):
            final_stack = self.stacks[player] + payouts[player]
            chip_delta = final_stack - self.starting_stack
            self.last_terminal_utilities.append(self._normalize_reward(chip_delta))

        total_final_chips = sum(self.stacks[player] + payouts[player] for player in range(self.num_players))
        if abs(total_final_chips - total_starting_chips) > 1e-6:
            raise RuntimeError("Chip conservation violated at terminal state")

        utility_sum = float(sum(self.last_terminal_utilities))
        if abs(utility_sum) > 1e-6:
            raise RuntimeError("Terminal utilities must sum to zero")

        info = {
            "terminal_utilities": self.last_terminal_utilities,
            "winners": winners,
            "reward_unit": self.reward_unit,
        }
        return self._get_state(), self.last_terminal_utilities[self.reference_player], True, info

    def _normalize_reward(self, chip_delta: float) -> float:
        if self.reward_unit == "bb":
            return chip_delta / self.big_blind
        return chip_delta

    def _active_player_count(self) -> int:
        return sum(self.active)

    def _active_players_with_chips(self) -> int:
        return sum(1 for player in range(self.num_players) if self.active[player] and self.stacks[player] > 0)
    
    def clone(self) -> 'PokerEnv':
        """
        Copia rápida del entorno. 10-50x más rápido que copy.deepcopy().
        Ideal para el recorrido del árbol en CFR y self-play.
        """
        new = object.__new__(PokerEnv)
        
        # --- Campos inmutables o compartidos seguros (referencia directa) ---
        new.num_players = self.num_players
        new.starting_stack = self.starting_stack
        new.small_blind = self.small_blind
        new.big_blind = self.big_blind
        new.reward_unit = self.reward_unit
        new.seed = self.seed
        new.rng = self.rng
        new.bet_sizing = self.bet_sizing
        new.ACTIONS = self.ACTIONS
        new.position = self.position  # Seguro compartirlo en traversal (el botón no muta mid-hand)
        new.hand_index = self.hand_index
        new.reference_player = self.reference_player

        # --- Campos mutables durante la mano (requieren copia superficial) ---
        new.deck = self.deck[:]
        new.hands = [h[:] for h in self.hands]
        new.board = self.board[:]
        
        new.pot = self.pot
        new.contributions = self.contributions[:]
        new.bets = self.bets[:]
        new.stacks = self.stacks[:]
        new.active = self.active[:]
        
        new.street = self.street
        new.history = self.history[:]
        new.done = self.done
        new.pending_players = self.pending_players[:]
        new.current_player = self.current_player
        
        new.last_raise_size = self.last_raise_size
        new.last_aggressor = self.last_aggressor
        new.last_action = self.last_action
        new.street_actions = self.street_actions
        new.last_terminal_utilities = self.last_terminal_utilities[:]
        new.last_winners = self.last_winners[:]
        
        return new
