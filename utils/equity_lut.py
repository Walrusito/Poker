"""
Lookup table for hand equity with bucketed postflop keys.

Preflop uses 169 hand classes per player count.
Postflop uses structural buckets instead of exact-card keys.
"""

import json
import pickle
import threading
import time
from pathlib import Path

from env.deck import Deck
from utils.hand_equity import HandEquity

_SAVE_INTERVAL = 50


class EquityLUT:
    def __init__(
        self,
        lut_dir="data/lut",
        mc_simulations=200,
        lut_simulations=1500,
        seed=None,
        use_torch_backend: bool = False,
        torch_device: str = "cuda",
    ):
        self.lut_dir = Path(lut_dir)
        self.lut_dir.mkdir(parents=True, exist_ok=True)

        self.preflop_path = self.lut_dir / "preflop_equity_lut.json"
        self.flop_path = self.lut_dir / "flop_equity_lut_v2.json"
        self.turn_path = self.lut_dir / "turn_equity_lut_v2.json"
        self.river_path = self.lut_dir / "river_equity_lut_v2.json"
        self.flop_legacy_path = self.lut_dir / "flop_equity_lut.json"
        self.turn_legacy_path = self.lut_dir / "turn_equity_lut.json"
        self.river_legacy_path = self.lut_dir / "river_equity_lut.json"

        self.preflop_table = self._load_table(self.preflop_path)
        self.flop_table = self._load_table(self.flop_path, fallback=self.flop_legacy_path)
        self.turn_table = self._load_table(self.turn_path, fallback=self.turn_legacy_path)
        self.river_table = self._load_table(self.river_path, fallback=self.river_legacy_path)

        self._lock = threading.RLock()
        self._pending_saves = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "fallback_calls": 0,
            "hits_by_street": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
            "misses_by_street": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
            "fallback_by_street": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
            "equity_time_ms_total": 0.0,
            "equity_calls": 0,
        }

        self.fallback_equity = HandEquity(
            simulations=mc_simulations,
            seed=seed,
            use_torch_backend=use_torch_backend,
            torch_device=torch_device,
        )
        self.lut_equity = HandEquity(
            simulations=lut_simulations,
            seed=seed,
            use_torch_backend=use_torch_backend,
            torch_device=torch_device,
        )
        self._ensure_table_files()

    def warmup_preflop(self, all_cards, max_players: int = 4):
        print(f"\n[LUT] Pre-calentando la LUT Preflop (2 hasta {max_players} jugadores)...")

        seen_classes = set()
        representative_hands = []
        for first_idx in range(len(all_cards)):
            for second_idx in range(first_idx + 1, len(all_cards)):
                hand = [all_cards[first_idx], all_cards[second_idx]]
                hand_class = self.preflop_key(hand, 2).split(":", 1)[1]
                if hand_class in seen_classes:
                    continue
                seen_classes.add(hand_class)
                representative_hands.append(hand)

        print(f"[LUT] Detectadas {len(representative_hands)} clases unicas de manos iniciales.")

        for num_players in range(2, max_players + 1):
            started = time.perf_counter()
            missing_count = 0

            for hand in representative_hands:
                key = self.preflop_key(hand, num_players)
                with self._lock:
                    exists = key in self.preflop_table
                if exists:
                    continue

                missing_count += 1
                equity = self.lut_equity.estimate(hand, [], num_players=num_players)
                with self._lock:
                    self.preflop_table[key] = equity
                    self._pending_saves["preflop"] += 1

            elapsed = time.perf_counter() - started
            print(f"[LUT] {num_players} jugadores: Calculadas {missing_count} nuevas llaves ({elapsed:.1f}s).")

        self.flush()
        print("[LUT] Pre-calentamiento Preflop completado y guardado en disco.\n")

    def estimate(self, hand, board=None, num_players: int = 2):
        if board is None:
            board = []

        board_len = len(board)
        if board_len == 0:
            street, key, table = "preflop", self.preflop_key(hand, num_players), self.preflop_table
        elif board_len == 3:
            street, key, table = "flop", self.flop_key(hand, board, num_players), self.flop_table
        elif board_len == 4:
            street, key, table = "turn", self.turn_key(hand, board, num_players), self.turn_table
        elif board_len == 5:
            street, key, table = "river", self.river_key(hand, board, num_players), self.river_table
        else:
            started = time.perf_counter()
            equity = self.fallback_equity.estimate(hand, board, num_players=num_players)
            with self._lock:
                self._stats["fallback_calls"] += 1
                self._stats["equity_time_ms_total"] += (time.perf_counter() - started) * 1000.0
                self._stats["equity_calls"] += 1
            return equity

        started = time.perf_counter()
        with self._lock:
            if key in table:
                self._stats["hits"] += 1
                self._stats["hits_by_street"][street] += 1
                self._stats["equity_time_ms_total"] += (time.perf_counter() - started) * 1000.0
                self._stats["equity_calls"] += 1
                return table[key]

        equity = self.lut_equity.estimate(hand, board, num_players=num_players)

        with self._lock:
            if key not in table:
                table[key] = equity
                self._stats["misses"] += 1
                self._stats["misses_by_street"][street] += 1
                self._pending_saves[street] += 1
                if self._pending_saves[street] >= _SAVE_INTERVAL:
                    self._save_street_table(street, table)
                    self._pending_saves[street] = 0
            else:
                self._stats["hits"] += 1
                self._stats["hits_by_street"][street] += 1
                equity = table[key]

            self._stats["equity_time_ms_total"] += (time.perf_counter() - started) * 1000.0
            self._stats["equity_calls"] += 1

        return equity

    def flush(self):
        with self._lock:
            for street, table in (
                ("preflop", self.preflop_table),
                ("flop", self.flop_table),
                ("turn", self.turn_table),
                ("river", self.river_table),
            ):
                if self._pending_saves[street] > 0:
                    self._save_street_table(street, table)
                    self._pending_saves[street] = 0

    def reset_stats(self):
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "fallback_calls": 0,
                "hits_by_street": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
                "misses_by_street": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
                "fallback_by_street": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
                "equity_time_ms_total": 0.0,
                "equity_calls": 0,
            }

    def get_stats(self):
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = float(self._stats["hits"] / total) if total > 0 else 0.0
            street_hit_rates = {}
            for street in ("preflop", "flop", "turn", "river"):
                street_total = self._stats["hits_by_street"][street] + self._stats["misses_by_street"][street]
                street_hit_rates[street] = (
                    float(self._stats["hits_by_street"][street] / street_total)
                    if street_total > 0 else 0.0
                )
            avg_ms = (
                float(self._stats["equity_time_ms_total"] / self._stats["equity_calls"])
                if self._stats["equity_calls"] > 0 else 0.0
            )
            return {
                "lut_hits": int(self._stats["hits"]),
                "lut_misses": int(self._stats["misses"]),
                "lut_hit_rate": hit_rate,
                "lut_fallback_calls": int(self._stats["fallback_calls"]),
                "lut_hit_rate_preflop": street_hit_rates["preflop"],
                "lut_hit_rate_flop": street_hit_rates["flop"],
                "lut_hit_rate_turn": street_hit_rates["turn"],
                "lut_hit_rate_river": street_hit_rates["river"],
                "lut_fallback_preflop": int(self._stats["fallback_by_street"]["preflop"]),
                "lut_fallback_flop": int(self._stats["fallback_by_street"]["flop"]),
                "lut_fallback_turn": int(self._stats["fallback_by_street"]["turn"]),
                "lut_fallback_river": int(self._stats["fallback_by_street"]["river"]),
                "equity_avg_ms": avg_ms,
            }

    def preflop_key(self, hand, num_players: int) -> str:
        cards = sorted(hand, key=lambda card: (Deck.rank(card), Deck.suit(card)), reverse=True)
        ranks_str = "23456789TJQKA"
        first_rank = ranks_str[Deck.rank(cards[0])]
        second_rank = ranks_str[Deck.rank(cards[1])]
        first_suit = Deck.suit(cards[0])
        second_suit = Deck.suit(cards[1])

        if first_rank == second_rank:
            hand_class = f"{first_rank}{second_rank}"
        else:
            suited = "s" if first_suit == second_suit else "o"
            if ranks_str.index(first_rank) < ranks_str.index(second_rank):
                first_rank, second_rank = second_rank, first_rank
            hand_class = f"{first_rank}{second_rank}{suited}"
        return f"{num_players}p:{hand_class}"

    def flop_key(self, hand, board, num_players: int) -> str:
        return f"{num_players}p:{self._hand_bucket(hand)}|{self._board_texture(board[:3])}"

    def turn_key(self, hand, board, num_players: int) -> str:
        river_bucket = self._rank_bucket(Deck.rank(board[3]))
        return f"{num_players}p:{self._hand_bucket(hand)}|{self._board_texture(board[:3])}-{river_bucket}"

    def river_key(self, hand, board, num_players: int) -> str:
        turn_bucket = self._rank_bucket(Deck.rank(board[3]))
        river_bucket = self._rank_bucket(Deck.rank(board[4]))
        return f"{num_players}p:{self._hand_bucket(hand)}|{self._board_texture(board[:3])}-{turn_bucket}{river_bucket}"

    @staticmethod
    def _rank_bucket(rank: int) -> int:
        if rank <= 5:
            return 0
        if rank <= 9:
            return 1
        return 2

    def _hand_bucket(self, hand) -> str:
        rank_buckets = sorted((self._rank_bucket(Deck.rank(card)) for card in hand), reverse=True)
        suited = "s" if Deck.suit(hand[0]) == Deck.suit(hand[1]) else "o"
        return f"h{rank_buckets[0]}-{rank_buckets[1]}{suited}"

    def _board_texture(self, board) -> str:
        if not board:
            return "b3TUL"

        suits = [Deck.suit(card) for card in board]
        ranks = sorted(Deck.rank(card) for card in board)
        suit_char = str(min(len(set(suits)), 3))
        spread = ranks[-1] - ranks[0]
        spread_char = "T" if spread <= 4 else ("M" if spread <= 8 else "W")
        paired_char = "P" if len(set(ranks)) < len(ranks) else "U"
        high_char = "H" if any(rank >= 10 for rank in ranks) else "L"
        return f"b{suit_char}{spread_char}{paired_char}{high_char}"

    @staticmethod
    def _load_table(path: Path, fallback: Path = None) -> dict:
        pkl_path = path.with_suffix(".pkl")
        if pkl_path.exists():
            with pkl_path.open("rb") as fh:
                return pickle.load(fh)

        target = path
        if not target.exists() and fallback is not None and fallback.exists():
            target = fallback
        if not target.exists():
            return {}
        with target.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _save_table(path: Path, table: dict):
        pkl_path = path.with_suffix(".pkl")
        tmp = pkl_path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(table, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pkl_path)

        json_tmp = path.with_suffix(".json.tmp")
        with json_tmp.open("w", encoding="utf-8") as fh:
            json.dump(table, fh, indent=2, sort_keys=True)
        json_tmp.replace(path)

    def _street_paths(self, street: str):
        if street == "preflop":
            return (self.preflop_path,)
        if street == "flop":
            return (self.flop_path, self.flop_legacy_path)
        if street == "turn":
            return (self.turn_path, self.turn_legacy_path)
        if street == "river":
            return (self.river_path, self.river_legacy_path)
        raise ValueError(f"Unknown street for LUT persistence: {street}")

    def _save_street_table(self, street: str, table: dict):
        for path in self._street_paths(street):
            self._save_table(path, table)

    def _ensure_table_files(self):
        for street, table in (
            ("preflop", self.preflop_table),
            ("flop", self.flop_table),
            ("turn", self.turn_table),
            ("river", self.river_table),
        ):
            for path in self._street_paths(street):
                if not path.exists():
                    self._save_table(path, table)
