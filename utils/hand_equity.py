import hashlib
import itertools
import threading
from collections import OrderedDict

import numpy as np
import torch

from env.deck import Deck
from env.rules import evaluate_7, evaluate_7_batch


class HandEquity:
    """
    Monte Carlo / exact river equity estimator against random opponents.
    Optimizado con NumPy para generacion vectorizada de repartos.
    """

    def __init__(self, simulations=200, seed=None, use_torch_backend: bool = False,
                 torch_device: str = "cuda", max_cache_size: int = 50_000):
        self.simulations = simulations
        self.seed = seed
        self.street_simulations = {
            0: simulations,
            3: simulations,
            4: max(simulations * 2, simulations),
            5: max(simulations * 3, simulations)
        }
        self.max_cache_size = max_cache_size
        self.cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self.use_torch_backend = bool(use_torch_backend)
        self.torch_device = torch_device

    def _query_seed(self, hand, board, num_players, simulations) -> int:
        payload = bytearray()
        payload.extend(str(self.seed).encode("utf-8"))
        payload.extend(b"|")
        payload.extend(bytes(sorted(int(card) for card in hand)))
        payload.extend(b"|")
        payload.extend(bytes(int(card) for card in board))
        payload.extend(f"|{int(num_players)}|{int(simulations)}".encode("utf-8"))
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    def estimate(self, hand, board=None, num_players: int = 2):
        if board is None:
            board = []

        if not 2 <= num_players <= 9:
            raise ValueError("num_players must be between 2 and 9")

        board_len = len(board)
        simulations = int(self.street_simulations.get(board_len, self.simulations))
        key = (tuple(sorted(hand)), tuple(sorted(board)), num_players, simulations)

        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

        if len(board) == 5 and num_players == 2:
            equity = self._estimate_exact_river_heads_up(hand, board)
        else:
            if self.use_torch_backend:
                equity = self._estimate_monte_carlo_torch(hand, board, num_players, simulations)
            else:
                equity = self._estimate_monte_carlo(hand, board, num_players, simulations)

        with self._lock:
            self.cache[key] = equity
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)
        return equity

    def _estimate_monte_carlo(self, hand, board, num_players, simulations):
        wins = 0.0
        ties = 0.0

        known = set(hand + board)
        available = np.array([c for c in range(52) if c not in known], dtype=np.int32)
        
        opponents = num_players - 1
        board_needed = max(0, 5 - len(board))
        cards_per_sim = (opponents * 2) + board_needed
        
        if len(available) < cards_per_sim:
            return 0.5

        local_rng = np.random.default_rng(self._query_seed(hand, board, num_players, simulations))
        n_avail = len(available)
        perms = np.empty((simulations, cards_per_sim), dtype=np.int32)
        for i in range(simulations):
            idx = local_rng.choice(n_avail, size=cards_per_sim, replace=False)
            perms[i] = available[idx]
        all_deals = perms

        batch_size = 128  # Aumentado para mejor aprovechamiento de evaluate_7_batch
        for batch_start in range(0, simulations, batch_size):
            local_sims = min(batch_size, simulations - batch_start)
            batch_deals = all_deals[batch_start:batch_start + local_sims]
            
            run_boards = []
            run_opponents = []

            for i in range(local_sims):
                draw = batch_deals[i]
                
                # Completar board
                rem_board = draw[:board_needed].tolist()
                full_board = board + rem_board
                
                # Repartir a oponentes
                opp_draw = draw[board_needed:]
                opponent_hands = [opp_draw[j*2 : (j+1)*2].tolist() for j in range(opponents)]
                
                run_boards.append(full_board)
                run_opponents.append(opponent_hands)

            # Evaluación por lotes
            hero_inputs = [hand + b for b in run_boards]
            hero_scores = evaluate_7_batch(hero_inputs)

            field_inputs = []
            for b, opps in zip(run_boards, run_opponents):
                for opp in opps:
                    field_inputs.append(opp + b)
            
            field_scores = evaluate_7_batch(field_inputs) if field_inputs else []

            cursor = 0
            for i in range(local_sims):
                hero_score = hero_scores[i]
                opp_count = len(run_opponents[i])
                run_field = field_scores[cursor : cursor + opp_count]
                cursor += opp_count

                best_field_score = max(run_field) if run_field else -1
                
                if hero_score > best_field_score:
                    wins += 1.0
                elif hero_score == best_field_score:
                    # Contar cuántos oponentes empatan con el mejor (que es igual al hero)
                    winners = 1 + sum(1 for s in run_field if s == hero_score)
                    ties += 1.0 / winners

        return (wins + ties) / simulations

    def _estimate_exact_river_heads_up(self, hand, board):
        known = set(hand + board)
        remaining = [card for card in range(52) if card not in known]

        hero_score = evaluate_7(hand + board)
        wins = 0.0
        ties = 0.0

        combos = list(itertools.combinations(remaining, 2))
        inputs = [list(opp_hand) + board for opp_hand in combos]
        scores = evaluate_7_batch(inputs)
        
        for opp_score in scores:
            if hero_score > opp_score:
                wins += 1.0
            elif hero_score == opp_score:
                ties += 0.5

        return (wins + ties) / len(combos) if combos else 0.0

    def _estimate_monte_carlo_torch(self, hand, board, num_players, simulations):
        # Mantenemos la lógica de Torch pero optimizada en flujo
        known = set(hand + board)
        available = [card for card in range(52) if card not in known]
        if not available: return 0.0

        try:
            device = self.torch_device if torch.cuda.is_available() and self.torch_device.startswith("cuda") else "cpu"
            gen = torch.Generator(device=device)
            gen.manual_seed(self._query_seed(hand, board, num_players, simulations))
            available_tensor = torch.tensor(available, dtype=torch.int64, device=device)
        except Exception:
            return self._estimate_monte_carlo(hand, board, num_players, simulations)

        opponents = num_players - 1
        board_needed = max(0, 5 - len(board))
        cards_needed = opponents * 2 + board_needed
        
        wins = 0.0
        ties = 0.0
        batch_size = 128

        for batch_start in range(0, simulations, batch_size):
            local_sims = min(batch_size, simulations - batch_start)
            
            run_boards = []
            run_opponents = []
            
            for _ in range(local_sims):
                perm = torch.randperm(len(available_tensor), generator=gen, device=device)[:cards_needed]
                draw = available_tensor[perm].tolist()
                
                opp_hands = [draw[j*2 : (j+1)*2] for j in range(opponents)]
                rem_board = draw[opponents*2:]
                full_board = board + rem_board
                
                run_boards.append(full_board)
                run_opponents.append(opp_hands)

            hero_scores = evaluate_7_batch([hand + b for b in run_boards])
            field_inputs = [opp + b for b, opps in zip(run_boards, run_opponents) for opp in opps]
            field_scores = evaluate_7_batch(field_inputs) if field_inputs else []

            cursor = 0
            for i in range(local_sims):
                h_score = hero_scores[i]
                o_count = len(run_opponents[i])
                f_scores = field_scores[cursor : cursor + o_count]
                cursor += o_count
                
                max_f = max(f_scores) if f_scores else -1
                if h_score > max_f:
                    wins += 1.0
                elif h_score == max_f:
                    winners = 1 + sum(1 for s in f_scores if s == h_score)
                    ties += 1.0 / winners

        return (wins + ties) / simulations
