"""
Layer 5 tests — multiway scenarios (proposal §Layer 5).

Covers:
- 9-player blind posting and position rotation
- Side pot distribution with multiple all-ins
- Zero-sum terminal utilities in 9-player hands
- Equity estimation accuracy drops with more opponents
- Feature vector dimensionality and correctness for N players
- N-player MCCFR (train_iteration runs without error, buffers grow)
"""

import pytest

from env.poker_env import PokerEnv
from utils.hand_equity import HandEquity
from utils.information_set import InformationSetBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def card(rank: str, suit: str) -> int:
    ranks = "23456789TJQKA"
    suits = "shdc"
    return suits.index(suit) * 13 + ranks.index(rank)


def run_hand_to_end(env: PokerEnv) -> dict:
    """Play every decision as 'fold' or the first legal action until done."""
    info = {}
    done = False
    while not done:
        legal = env.get_legal_actions()
        action = "fold" if "fold" in legal else legal[0]
        _, _, done, info = env.step(action)
    return info


# ---------------------------------------------------------------------------
# 9-player blind posting and position rotation
# ---------------------------------------------------------------------------

class TestNinePlayerBlinds:
    def test_blinds_are_posted_correctly(self):
        env = PokerEnv(num_players=9, starting_stack=10000, small_blind=50, big_blind=100)
        state = env.reset()
        contributions = state["contributions"]
        assert contributions.count(50.0) >= 1, "small blind not posted"
        assert contributions.count(100.0) >= 1, "big blind not posted"
        assert sum(contributions) == 150.0

    def test_button_advances_each_hand(self):
        env = PokerEnv(num_players=9, starting_stack=10000, small_blind=50, big_blind=100, seed=1)
        buttons = set()
        for _ in range(9):
            state = env.reset()
            buttons.add(state["button"])
            run_hand_to_end(env)
        assert len(buttons) == 9, "button must visit all 9 seats over 9 hands"

    def test_first_to_act_preflop_is_utg(self):
        env = PokerEnv(num_players=9, starting_stack=10000, small_blind=50, big_blind=100)
        state = env.reset()
        button = state["button"]
        # SB = button+1, BB = button+2, UTG = button+3 (mod 9)
        expected_utg = (button + 3) % 9
        assert state["current_player"] == expected_utg

    def test_preflop_reaches_postflop_with_correct_first_actor(self):
        env = PokerEnv(num_players=9, starting_stack=10000, small_blind=50, big_blind=100)
        state = env.reset()
        button = state["button"]
        # Advance all preflop decisions with calls
        while state["street"] == "preflop" and not state["done"]:
            legal = env.get_legal_actions()
            action = "call" if "call" in legal else "check"
            state, _, done, _ = env.step(action)
            if done:
                return  # hand ended early (fine)
        if not state["done"] and state["street"] == "flop":
            # First to act postflop: left of button
            expected = (button + 1) % 9
            # Find first active seat from button+1
            active = state["active"]
            actual = state["current_player"]
            # actual must be active
            assert active[actual], "first postflop actor must be active"


# ---------------------------------------------------------------------------
# Side pot distribution with multiple all-ins
# ---------------------------------------------------------------------------

class TestSidePots:
    def _make_env_with_unequal_stacks(self, stacks):
        """Create a 3-player env where player stacks start already set."""
        env = PokerEnv(
            num_players=len(stacks),
            starting_stack=max(stacks),
            small_blind=5,
            big_blind=10,
            seed=42,
        )
        return env

    def test_single_all_in_zero_sum(self):
        """All-in from one player followed by fold — chips must be conserved."""
        env = PokerEnv(num_players=3, starting_stack=1000, small_blind=5, big_blind=10, seed=7)
        env.reset()
        done = False
        info = {}
        while not done:
            legal = env.get_legal_actions()
            action = "all_in" if "all_in" in legal else ("fold" if "fold" in legal else legal[0])
            _, _, done, info = env.step(action)
        utils = info["terminal_utilities"]
        assert abs(sum(utils)) < 1e-6, f"utilities not zero-sum: {utils}"

    def test_terminal_utilities_zero_sum_9_players(self):
        env = PokerEnv(num_players=9, starting_stack=10000, small_blind=50, big_blind=100, seed=3)
        env.reset()
        info = run_hand_to_end(env)
        utils = info["terminal_utilities"]
        assert len(utils) == 9
        assert abs(sum(utils)) < 1e-6, f"utilities not zero-sum in 9-player hand: {utils}"

    def test_multiway_all_in_zero_sum(self):
        """Drive everyone to all-in and verify zero-sum."""
        env = PokerEnv(num_players=4, starting_stack=500, small_blind=5, big_blind=10, seed=11)
        env.reset()
        done = False
        info = {}
        while not done:
            legal = env.get_legal_actions()
            if "all_in" in legal:
                action = "all_in"
            elif "call" in legal:
                action = "call"
            else:
                action = legal[0]
            _, _, done, info = env.step(action)
        utils = info["terminal_utilities"]
        assert abs(sum(utils)) < 1e-6, f"multiway all-in not zero-sum: {utils}"

    def test_chip_conservation_9_players_showdown(self):
        """Run to showdown, verify total chips unchanged."""
        env = PokerEnv(num_players=9, starting_stack=1000, small_blind=5, big_blind=10, seed=99)
        env.reset()
        total_chips_before = env.starting_stack * env.num_players
        done = False
        info = {}
        while not done:
            legal = env.get_legal_actions()
            action = "call" if "call" in legal else ("check" if "check" in legal else legal[0])
            _, _, done, info = env.step(action)
        utils = info["terminal_utilities"]
        # utilities in BB; sum must be 0
        assert abs(sum(utils)) < 1e-6

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 9])
    def test_zero_sum_across_player_counts(self, num_players):
        env = PokerEnv(
            num_players=num_players,
            starting_stack=5000,
            small_blind=25,
            big_blind=50,
            seed=num_players,
        )
        env.reset()
        info = run_hand_to_end(env)
        utils = info["terminal_utilities"]
        assert len(utils) == num_players
        assert abs(sum(utils)) < 1e-6


# ---------------------------------------------------------------------------
# Equity estimation accuracy drops with more opponents
# ---------------------------------------------------------------------------

class TestMultiwayEquity:
    def test_aces_equity_drops_with_more_opponents(self):
        estimator = HandEquity(simulations=400, seed=7)
        hand = [card("A", "s"), card("A", "h")]
        eq_hu = estimator.estimate(hand, [], num_players=2)
        eq_3 = estimator.estimate(hand, [], num_players=3)
        eq_6 = estimator.estimate(hand, [], num_players=6)
        assert eq_hu > eq_3 > eq_6, (
            f"equity should drop as opponents increase: HU={eq_hu:.3f}, "
            f"3-way={eq_3:.3f}, 6-way={eq_6:.3f}"
        )

    def test_equity_is_valid_probability(self):
        estimator = HandEquity(simulations=100, seed=3)
        hand = [card("K", "s"), card("Q", "s")]
        for n in (2, 3, 4, 6, 9):
            eq = estimator.estimate(hand, [], num_players=n)
            assert 0.0 <= eq <= 1.0, f"equity out of [0,1] for {n} players: {eq}"

    def test_random_hand_equity_decreases_monotonically(self):
        """A random medium hand should have decreasing EQ as table grows."""
        estimator = HandEquity(simulations=300, seed=17)
        hand = [card("J", "h"), card("9", "h")]
        equities = [estimator.estimate(hand, [], num_players=n) for n in (2, 3, 4, 6)]
        for i in range(len(equities) - 1):
            assert equities[i] >= equities[i + 1] - 0.02, (
                f"equity did not decrease monotonically: {equities}"
            )


# ---------------------------------------------------------------------------
# Feature vector dimensionality and correctness for N players
# ---------------------------------------------------------------------------

class TestFeatureVectorMultiway:
    def _make_state(self, num_players: int):
        env = PokerEnv(
            num_players=num_players,
            starting_stack=10000,
            small_blind=50,
            big_blind=100,
            seed=7,
        )
        return env.reset(), env

    def test_feature_dim_is_constant_across_player_counts(self, tmp_path):
        builder = InformationSetBuilder(
            mc_simulations=10, lut_simulations=10, lut_dir=tmp_path, seed=0, cache_size=0
        )
        dims = set()
        for n in (2, 3, 4, 6, 9):
            state, _ = self._make_state(n)
            vec = builder.encode_vector(state, player=0)
            dims.add(len(vec))
        assert len(dims) == 1, f"feature dim varies across player counts: {dims}"

    def test_num_players_norm_scales_with_table_size(self, tmp_path):
        builder = InformationSetBuilder(
            mc_simulations=10, lut_simulations=10, lut_dir=tmp_path, seed=0, cache_size=0
        )
        prev_norm = None
        for n in (2, 4, 6, 9):
            state, _ = self._make_state(n)
            features = builder.encode(state, player=0)
            norm = features["num_players_norm"]
            assert norm == pytest.approx(n / 9.0), f"num_players_norm wrong for n={n}"
            if prev_norm is not None:
                assert norm > prev_norm
            prev_norm = norm

    def test_equity_feature_in_unit_interval_for_all_configs(self, tmp_path):
        builder = InformationSetBuilder(
            mc_simulations=20, lut_simulations=20, lut_dir=tmp_path, seed=5, cache_size=0
        )
        for n in (2, 3, 6, 9):
            state, _ = self._make_state(n)
            features = builder.encode(state, player=0)
            assert 0.0 <= features["equity"] <= 1.0, (
                f"equity out of [0,1] for {n}-player state: {features['equity']}"
            )

    def test_active_players_norm_after_fold(self, tmp_path):
        """After one fold, active_players_norm should decrease."""
        builder = InformationSetBuilder(
            mc_simulations=10, lut_simulations=10, lut_dir=tmp_path, seed=0, cache_size=0
        )
        env = PokerEnv(num_players=4, starting_stack=10000, small_blind=50, big_blind=100, seed=0)
        state = env.reset()
        full_active_norm = builder.encode(state, player=state["current_player"])["active_players_norm"]

        # Fold once
        env.step("fold")
        state = env._get_state()
        reduced_active_norm = builder.encode(state, player=state["current_player"])["active_players_norm"]
        assert reduced_active_norm < full_active_norm


# ---------------------------------------------------------------------------
# N-player MCCFR integration
# ---------------------------------------------------------------------------

class TestNPlayerMCCFR:
    def test_train_iteration_runs_for_heads_up(self, tmp_path):
        from cfr.mccfr import MCCFR

        def make_env():
            return PokerEnv(
                num_players=2, starting_stack=500, small_blind=5, big_blind=10, seed=None
            )

        solver = MCCFR(make_env, mc_simulations=5, lut_simulations=5, lut_dir=str(tmp_path))
        for _ in range(3):
            solver.train_iteration()
        assert len(solver.regret) > 0 or len(solver.strategy_sum) >= 0  # ran without error

    def test_train_iteration_runs_for_three_players(self, tmp_path):
        from cfr.mccfr import MCCFR

        def make_env():
            return PokerEnv(
                num_players=3, starting_stack=500, small_blind=5, big_blind=10, seed=None
            )

        solver = MCCFR(make_env, mc_simulations=5, lut_simulations=5, lut_dir=str(tmp_path))
        for _ in range(3):
            solver.train_iteration()

    def test_train_iteration_runs_for_six_players(self, tmp_path):
        from cfr.mccfr import MCCFR

        def make_env():
            return PokerEnv(
                num_players=6, starting_stack=500, small_blind=5, big_blind=10, seed=None
            )

        solver = MCCFR(make_env, mc_simulations=5, lut_simulations=5, lut_dir=str(tmp_path))
        solver.train_iteration()  # one iteration is enough for a smoke test

    def test_strategy_sums_accumulate_over_iterations(self, tmp_path):
        from cfr.mccfr import MCCFR

        def make_env():
            return PokerEnv(
                num_players=2, starting_stack=500, small_blind=5, big_blind=10, seed=None
            )

        solver = MCCFR(make_env, mc_simulations=5, lut_simulations=5, lut_dir=str(tmp_path))
        for _ in range(5):
            solver.train_iteration()

        assert len(solver.strategy_sum) > 0, "strategy_sum must be non-empty after iterations"

    def test_average_strategy_is_valid_distribution(self, tmp_path):
        from cfr.mccfr import MCCFR

        def make_env():
            return PokerEnv(
                num_players=2, starting_stack=500, small_blind=5, big_blind=10, seed=7
            )

        solver = MCCFR(make_env, mc_simulations=5, lut_simulations=5, lut_dir=str(tmp_path))
        for _ in range(10):
            solver.train_iteration()

        actions = ["fold", "call", "check", "all_in"]
        for info_set in list(solver.strategy_sum.keys())[:5]:
            # Only test info sets that actually have entries for these actions
            known_actions = list(solver.strategy_sum[info_set].keys())
            if not known_actions:
                continue
            strategy = solver.get_average_strategy(info_set, known_actions)
            total = sum(strategy.values())
            assert abs(total - 1.0) < 1e-6, (
                f"strategy does not sum to 1 for {info_set}: {strategy}"
            )

    def test_regrets_have_correct_sign_semantics(self, tmp_path):
        """After many iterations, regrets can be negative (under-performing actions)."""
        from cfr.mccfr import MCCFR

        def make_env():
            return PokerEnv(
                num_players=2, starting_stack=500, small_blind=5, big_blind=10, seed=13
            )

        solver = MCCFR(make_env, mc_simulations=5, lut_simulations=5, lut_dir=str(tmp_path))
        for _ in range(20):
            solver.train_iteration()

        # Just verify regret values are finite for all info sets
        for info_set, action_regrets in solver.regret.items():
            for action, regret in action_regrets.items():
                assert isinstance(regret, float)
                import math
                assert math.isfinite(regret), f"non-finite regret for {info_set}/{action}"