"""
Monte Carlo CFR — N-player extension.

Replaces the original 2-player (p0, p1) reach-probability implementation with a
reach_probs vector of length N.  The rest of the algorithm stays identical:

  - counterfactual reach for player i  = product of all OTHER players' reach probs
  - own reach for strategy accumulation = reach_probs[i]
  - regret update uses counterfactual reach (standard outcome-sampling MCCFR)

Optimisations applied:
  - Step 1:  snapshot/restore instead of deepcopy in tree traversal
  - Step 9:  encode_tuple with fast hash instead of SHA256
  - Step 10: regret-based pruning (RBP) — skip action branches where all
             cumulative regrets are strongly negative (Bowling et al., 2015)

Reference: Lanctot et al. (2009), "Monte Carlo Sampling for Regret Minimization
in Extensive Games", NeurIPS.
"""

import math
from collections import defaultdict
from typing import List

from utils.information_set import InformationSetBuilder


class MCCFR:
    """
    Outcome-sampling Monte Carlo CFR for N-player NLHE.

    Parameters
    ----------
    env_class : callable
        Zero-argument factory that returns a fresh PokerEnv.
    mc_simulations : int
        MC simulations passed to InformationSetBuilder for equity estimates.
    lut_simulations : int
        Simulations used to populate the preflop/flop LUT.
    lut_dir : str
        Directory for equity LUT files.
    seed : int | None
        Optional RNG seed.
    prune_threshold : float
        Regret-based pruning threshold.  Actions whose cumulative regret
        falls below ``-prune_threshold`` are skipped during traversal.
        Set to 0 to disable pruning.
    prune_after : int
        Only enable pruning after this many iterations (regrets need time
        to stabilise).
    """

    def __init__(
        self,
        env_class,
        mc_simulations: int = 200,
        lut_simulations: int = 1500,
        lut_dir: str = "data/lut",
        seed=None,
        prune_threshold: float = 1000.0,
        prune_after: int = 25,
    ):
        self.env_class = env_class
        self.iss = InformationSetBuilder(
            mc_simulations=mc_simulations,
            lut_simulations=lut_simulations,
            lut_dir=lut_dir,
            seed=seed,
        )
        self.regret: dict = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: dict = defaultdict(lambda: defaultdict(float))
        self.prune_threshold = prune_threshold
        self.prune_after = prune_after
        self._iteration = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_iteration(self):
        """Run one CFR traversal of a freshly-dealt hand."""
        env = self.env_class()
        num_players = int(env.num_players)
        reach_probs = [1.0] * num_players
        self._cfr(env, reach_probs)
        self._iteration += 1

    def get_average_strategy(self, info_set: str, actions: List[str]) -> dict:
        """Return the time-averaged strategy (Nash approximation) for an info set."""
        strat = self.strategy_sum[info_set]
        total = sum(strat.get(a, 0.0) for a in actions)
        if total > 0:
            return {a: strat.get(a, 0.0) / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    # ------------------------------------------------------------------
    # Core recursive traversal
    # ------------------------------------------------------------------

    def _cfr(self, env, reach_probs: List[float]) -> List[float]:
        """
        Recursive CFR traversal.

        Returns
        -------
        List[float]
            Per-player utilities from this node onward.
        """
        state = env._get_state()

        # --- Terminal node ---
        if state["street"] == "showdown" or state.get("done", False):
            return list(env.get_terminal_utilities())

        player: int = state["current_player"]
        actions = env.get_legal_actions()
        if not actions:
            return list(env.get_terminal_utilities())

        info_set = self.iss.encode_tuple(state, player)
        strategy = self._get_strategy(info_set, actions)

        # Counterfactual reach: product of all players' reach probs except `player`.
        cf_reach = self._counterfactual_reach(reach_probs, player)

        # Accumulate strategy sum weighted by the acting player's own reach.
        own_reach = reach_probs[player]
        for action in actions:
            self.strategy_sum[info_set][action] += own_reach * strategy[action]

        # --- Regret-based pruning (Step 10) ---
        pruning_active = (
            self.prune_threshold > 0
            and self._iteration >= self.prune_after
        )

        # --- Evaluate each action (snapshot/restore instead of deepcopy) ---
        action_utils: dict[str, List[float]] = {}
        node_util = [0.0] * len(reach_probs)

        for action in actions:
            # RBP: skip actions with strongly negative regret
            if pruning_active:
                cumulative = self.regret[info_set].get(action, 0.0)
                if cumulative < -self.prune_threshold:
                    action_utils[action] = [0.0] * len(reach_probs)
                    continue

            snap = env.get_snapshot()
            next_state, reward, done, info = env.step(action)

            if done:
                action_utils[action] = list(env.get_terminal_utilities())
            else:
                new_reach = list(reach_probs)
                new_reach[player] *= strategy[action]
                action_utils[action] = self._cfr(env, new_reach)

            env.restore_snapshot(snap)

            for p in range(len(reach_probs)):
                node_util[p] += strategy[action] * action_utils[action][p]

        # --- Regret update (only for acting player, using counterfactual reach) ---
        for action in actions:
            regret = action_utils[action][player] - node_util[player]
            self.regret[info_set][action] += cf_reach * regret

        return node_util

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _counterfactual_reach(self, reach_probs: List[float], player: int) -> float:
        """Product of all players' reach probabilities except `player`."""
        result = 1.0
        for idx, rp in enumerate(reach_probs):
            if idx != player:
                result *= rp
        return result

    def _get_strategy(self, info_set: str, actions: List[str]) -> dict:
        """Regret matching: map positive regrets to a probability distribution."""
        regrets = self.regret[info_set]
        positive = {a: max(regrets[a], 0.0) for a in actions}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def _terminal_utility(self, env) -> float:
        """Convenience: P0 utility (kept for backward-compat callers)."""
        return env.get_terminal_utilities()[0]
