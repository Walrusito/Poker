from collections import OrderedDict
import hashlib
import struct
import threading

import numpy as np

from utils.card_abstraction import CardAbstraction
from utils.math_features import compute_effective_stack, compute_implied_odds, compute_pot_odds

_ZOBRIST_CARD = tuple(hash(("card", i)) for i in range(53))
_ZOBRIST_PLAYER = tuple(hash(("player", i)) for i in range(10))
_ZOBRIST_STREET = {"preflop": hash("preflop"), "flop": hash("flop"),
                   "turn": hash("turn"), "river": hash("river"),
                   "showdown": hash("showdown")}
_PACK_FLOAT = struct.Struct("!d")


class InformationSetBuilder:
    """
    Builds the engineered information-set features used by the policy/regret nets.

    The cache stores both the float32 vector used by the models and the full
    precision feature dictionary used by heuristics/tests.
    """

    FEATURE_SCHEMA_VERSION = 2
    FEATURE_KEYS = (
        "equity", "equity_bucket", "pot_odds", "implied_odds", "spr_norm",
        "pot_size_norm", "to_call_norm", "last_raise_norm", "street", "position",
        "aggressor_distance_norm", "active_players_norm", "players_to_act_norm",
        "num_players_norm", "has_aggressor", "facing_bet", "is_last_to_act",
        "is_button", "is_small_blind", "is_big_blind", "hero_stack_norm",
        "hero_contribution_norm", "board_paired", "board_monotone",
        "board_connected", "board_high_card_density", "blocker_ace",
        "blocker_king", "nut_potential", "aggression_last4_norm",
        "last_raise_to_pot_ratio",
    )

    def __init__(
        self,
        mc_simulations: int = 200,
        lut_simulations: int = 1500,
        lut_dir: str = "data/lut",
        seed=None,
        cache_size: int = 100000,
        use_torch_backend: bool = False,
        torch_device: str = "cuda",
    ):
        self.card_abs = CardAbstraction(
            mc_simulations=mc_simulations,
            lut_simulations=lut_simulations,
            lut_dir=lut_dir,
            seed=seed,
            use_torch_backend=use_torch_backend,
            torch_device=torch_device,
        )
        self.feature_dim = len(self.FEATURE_KEYS)
        self.cache_size = max(0, int(cache_size))
        self._vector_cache = OrderedDict()
        self._feature_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.RLock()

        self.feature_schema = {
            "version": self.FEATURE_SCHEMA_VERSION,
            "keys": list(self.FEATURE_KEYS),
            "feature_dim": self.feature_dim,
            "fingerprint": self.feature_schema_fingerprint(),
        }

    def feature_schema_fingerprint(self):
        payload = f"{self.FEATURE_SCHEMA_VERSION}|{'|'.join(self.FEATURE_KEYS)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        return {
            "feature_cache_hit_rate": self._cache_hits / max(1, total),
            "feature_cache_size": len(self._vector_cache),
        }

    @staticmethod
    def _round_values(values, digits: int = 4):
        return tuple(round(float(value), digits) for value in values)

    @staticmethod
    def _hash_float(h: int, value: float) -> int:
        return h ^ hash(_PACK_FLOAT.pack(round(value, 4)))

    def _state_key(self, state, player: int):
        if self.cache_size == 0:
            return None

        h = _ZOBRIST_PLAYER[player]
        h ^= _ZOBRIST_STREET.get(state.get("street"), 0)

        hand = state["hands"][player]
        for card in sorted(hand):
            h ^= _ZOBRIST_CARD[card]

        for card in state.get("board", []):
            h ^= _ZOBRIST_CARD[card] * 31

        h ^= hash(state.get("num_players", 2)) * 997

        active = state.get("active", [])
        active_bits = 0
        for i, flag in enumerate(active):
            if flag:
                active_bits |= (1 << i)
        h ^= active_bits * 7919

        h = self._hash_float(h, state.get("pot", 0.0))
        for v in state.get("bets", []):
            h = self._hash_float(h, v)
        for v in state.get("stacks", []):
            h = self._hash_float(h, v)
        for v in state.get("contributions", []):
            h = self._hash_float(h, v)
        h = self._hash_float(h, state.get("starting_stack", 0.0))
        h = self._hash_float(h, state.get("big_blind", 0.0))
        h = self._hash_float(h, state.get("last_raise_size", 0.0))

        h ^= hash(state.get("button")) * 1009
        h ^= hash(state.get("last_aggressor")) * 2003

        pending = state.get("pending_players", [])
        for i, seat in enumerate(pending):
            h ^= hash((i, seat)) * 4007

        for actor, action in state.get("history", [])[-4:]:
            h ^= hash((int(actor), action)) * 8009

        return h

    def _cache_lookup(self, cache, cache_key):
        with self._lock:
            cached = cache.get(cache_key)
            if cached is not None:
                cache.move_to_end(cache_key)
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            return cached

    def _store_cached_payload(self, cache_key, vector, feature_map):
        if cache_key is None:
            return

        with self._lock:
            self._vector_cache[cache_key] = vector
            self._vector_cache.move_to_end(cache_key)
            self._feature_cache[cache_key] = feature_map
            self._feature_cache.move_to_end(cache_key)

            while len(self._vector_cache) > self.cache_size:
                evicted_key, _ = self._vector_cache.popitem(last=False)
                self._feature_cache.pop(evicted_key, None)

            while len(self._feature_cache) > self.cache_size:
                evicted_key, _ = self._feature_cache.popitem(last=False)
                self._vector_cache.pop(evicted_key, None)

    def _build_feature_payload(self, state, player=0):
        hand = state["hands"][player]
        board = list(state.get("board", []))
        active_flags = [bool(flag) for flag in state.get("active", [])]
        num_active = sum(active_flags)
        num_players = int(state["num_players"])

        equity_players = max(2, num_active)
        equity = float(self.card_abs.estimate_equity(hand, board, num_players=equity_players))
        bucket = self.card_abs.bucket_hand(hand, board, num_players=equity_players)
        equity_bucket = float(bucket / (self.card_abs.num_buckets - 1))

        pot = float(state.get("pot", 0.0))
        bets = list(state.get("bets", []))
        to_call = float(max(bets) - bets[player]) if bets else 0.0
        hero_stack = float(state["stacks"][player])
        opponent_stacks = [
            float(state["stacks"][idx])
            for idx in range(num_players)
            if idx != player and active_flags[idx]
        ]
        eff_stack = float(compute_effective_stack(hero_stack, opponent_stacks, to_call))

        big_blind = float(state.get("big_blind", 100.0))
        starting_stack = float(state.get("starting_stack", 10000.0))
        blind_denom = big_blind if big_blind > 0.0 else 1.0
        stack_denom = starting_stack if starting_stack > 0.0 else 1.0
        stack_in_bb = stack_denom / blind_denom
        spr_denom = eff_stack + pot
        street_map = {"preflop": 0.0, "flop": 0.33, "turn": 0.66, "river": 1.0}
        pending_players = list(state.get("pending_players", []))
        contributions = list(state.get("contributions", [0.0] * num_players))
        last_raise_size = float(state.get("last_raise_size", big_blind))

        feature_values = [
            equity,
            equity_bucket,
            float(compute_pot_odds(pot, to_call)),
            float(compute_implied_odds(pot, to_call, eff_stack)),
            float(eff_stack / spr_denom) if spr_denom > 0.0 else 0.0,
            float((pot / blind_denom) / stack_in_bb) if stack_in_bb > 0.0 else 0.0,
            float((to_call / blind_denom) / stack_in_bb) if stack_in_bb > 0.0 else 0.0,
            float(last_raise_size / stack_denom),
            float(street_map.get(state["street"], 0.0)),
            float(self._relative_position(player, state.get("button", 0), num_players)),
            float(
                self._relative_position(player, int(state.get("last_aggressor", 0)), num_players)
                if state.get("last_aggressor") is not None else 0.0
            ),
            float(num_active / num_players) if num_players > 0 else 0.0,
            float(len(pending_players) / num_players) if num_players > 0 else 0.0,
            float(num_players / 9.0),
            1.0 if state.get("last_aggressor") is not None else 0.0,
            1.0 if to_call > 0.0 else 0.0,
            1.0 if pending_players and pending_players[-1] == player else 0.0,
            1.0 if player == state.get("button") else 0.0,
            1.0 if player == self._blind_positions(state.get("button", 0), num_players)[0] else 0.0,
            1.0 if player == self._blind_positions(state.get("button", 0), num_players)[1] else 0.0,
            float(hero_stack / stack_denom),
            float(contributions[player] / stack_denom),
        ]

        board_paired, board_monotone, board_connected, board_high = self._board_texture_features(board)
        blocker_ace, blocker_king, nut_potential = self._hero_blockers_and_nut_potential(hand, board)
        feature_values.extend(
            [
                float(board_paired),
                float(board_monotone),
                float(board_connected),
                float(board_high),
                float(blocker_ace),
                float(blocker_king),
                float(nut_potential),
                float(self._aggression_from_history(state.get("history", []))),
                float(last_raise_size / max(pot, blind_denom)),
            ]
        )

        vector = np.array(feature_values, dtype=np.float32)
        feature_map = {key: value for key, value in zip(self.FEATURE_KEYS, feature_values)}
        return vector, feature_map

    def encode_vector(self, state, player=0):
        cache_key = self._state_key(state, player)
        if cache_key is not None:
            cached_vector = self._cache_lookup(self._vector_cache, cache_key)
            if cached_vector is not None:
                return cached_vector

        vector, feature_map = self._build_feature_payload(state, player)
        self._store_cached_payload(cache_key, vector, feature_map)
        return vector

    @staticmethod
    def _relative_position(player: int, button: int, num_players: int) -> float:
        if num_players <= 1:
            return 0.0
        return ((player - button) % num_players) / (num_players - 1)

    @staticmethod
    def _blind_positions(button: int, num_players: int):
        sb = button if num_players == 2 else (button + 1) % num_players
        bb = (sb + 1) % num_players
        return sb, bb

    def _board_texture_features(self, board):
        if not board:
            return 0.0, 0.0, 0.0, 0.0

        ranks = sorted(card % 13 for card in board)
        suits = [card // 13 for card in board]
        unique_ranks = set(ranks)
        paired = 1.0 if len(unique_ranks) < len(ranks) else 0.0
        monotone = 1.0 if len(set(suits)) == 1 and len(board) >= 3 else 0.0
        connected = 1.0 if len(board) >= 3 and (max(ranks) - min(ranks) <= 4) else 0.0
        high_density = sum(1 for rank in ranks if rank >= 8) / len(board)
        return paired, monotone, connected, high_density

    def _hero_blockers_and_nut_potential(self, hand, board):
        hero_ranks = [card % 13 for card in hand]
        blocker_ace = 1.0 if 12 in hero_ranks else 0.0
        blocker_king = 1.0 if 11 in hero_ranks else 0.0

        all_cards = hand + board
        suits = [card // 13 for card in all_cards]
        max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)
        broadway_count = sum(1 for rank in hero_ranks if rank >= 8)
        nut_potential = min(1.0, 0.5 * float(max_suit_count >= 4) + 0.25 * broadway_count)
        return blocker_ace, blocker_king, nut_potential

    def _aggression_from_history(self, history):
        if not history:
            return 0.0
        recent = history[-4:]
        aggressive_actions = sum(
            1 for _, action in recent
            if "bet" in action or "raise" in action or "all_in" in action
        )
        return aggressive_actions / len(recent)

    def encode(self, state, player=0):
        cache_key = self._state_key(state, player)
        if cache_key is not None:
            cached_features = self._cache_lookup(self._feature_cache, cache_key)
            if cached_features is not None:
                return cached_features

        vector, feature_map = self._build_feature_payload(state, player)
        self._store_cached_payload(cache_key, vector, feature_map)
        return feature_map

    def encode_tuple(self, state, player=0, precision=4):
        vec = self.encode_vector(state, player)
        return tuple(round(float(value), precision) for value in vec)
