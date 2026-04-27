import copy
from collections import defaultdict

from utils.information_set import InformationSetBuilder


class MCCFR:

    def __init__(self, env_class):
        self.env_class = env_class

        self.iss = InformationSetBuilder()

        self.regret = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

    # -----------------------------
    # ENTRY
    # -----------------------------
    def train_iteration(self):
        env = self.env_class()
        if env.num_players != 2:
            raise NotImplementedError("Tabular MCCFR is only supported for heads-up environments")
        self._cfr(env, 1.0, 1.0)

    # -----------------------------
    # CORE CFR
    # -----------------------------
    def _cfr(self, env, p0, p1):

        state = env._get_state()

        if state["street"] == "showdown" or state.get("done", False):
            return self._terminal_utility(env)

        player = state["current_player"]

        info_set = self.iss.encode_tuple(state, player)

        # FIX: use env-native legal actions to avoid silent no-ops
        # when bet_sizing actions like "raise_0.5" were passed to step()
        # that only understood "fold"/"call"/"raise"
        actions = env.get_legal_actions()

        strategy = self._get_strategy(info_set, actions)

        node_util = 0.0
        util = {}

        # -----------------------------
        # DECISION NODE
        # -----------------------------
        for action in actions:

            env_copy = copy.deepcopy(env)

            # FIX: step() returns (state, reward, done, info) — 4 values
            # Previous code unpacked only 3, causing ValueError
            next_state, reward, done, _ = env_copy.step(action)

            if done:
                util[action] = reward
            else:
                util[action] = self._cfr(
                    env_copy,
                    p0 * (strategy[action] if player == 0 else 1),
                    p1 * (strategy[action] if player == 1 else 1)
                )

            node_util += strategy[action] * util[action]

        # -----------------------------
        # REGRET UPDATE
        # -----------------------------
        for action in actions:
            regret = util[action] - node_util if player == 0 else node_util - util[action]
            if player == 0:
                self.regret[info_set][action] += p1 * regret
            else:
                self.regret[info_set][action] += p0 * regret

        # -----------------------------
        # STRATEGY SUM — FIX: must be weighted by the current player's
        # own reach probability for the average strategy to converge
        # to Nash.  The original code used weight=1 (unweighted).
        # -----------------------------
        reach = p0 if player == 0 else p1
        for action in actions:
            self.strategy_sum[info_set][action] += reach * strategy[action]

        return node_util

    # -----------------------------
    # REGRET MATCHING
    # -----------------------------
    def _get_strategy(self, info_set, actions):
        regrets = self.regret[info_set]
        positive = {a: max(regrets[a], 0.0) for a in actions}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    # -----------------------------
    # AVERAGE STRATEGY  (Nash approximation — use this for play)
    # -----------------------------
    def get_average_strategy(self, info_set, actions):
        strat = self.strategy_sum[info_set]
        total = sum(strat.get(a, 0.0) for a in actions)
        if total > 0:
            return {a: strat.get(a, 0.0) / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    # -----------------------------
    # TERMINAL
    # -----------------------------
    def _terminal_utility(self, env):
        return env.get_terminal_utilities()[0]
