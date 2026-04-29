import random


class RangeModel:
    """
    Opponent range distribution model

    Representa creencia sobre manos del oponente.
    """

    def __init__(self):
        # simplified preflop range buckets
        self.range = self._init_range()

    def _init_range(self):
        return {
            "AA": 0.05,
            "KK": 0.05,
            "QQ": 0.08,
            "JJ": 0.08,
            "AK": 0.15,
            "AQ": 0.12,
            "AJ": 0.10,
            "KQ": 0.10,
            "random": 0.27
        }

    def sample_hand(self):
        """
        Sample opponent hand from belief distribution
        """
        keys = list(self.range.keys())
        probs = list(self.range.values())

        return random.choices(keys, probs)[0]

    def update(self, action, street):
        """
        Bayesian-like update (simplified heuristic)

        In real solvers:
        -> belief update via CFR regrets
        """

        if action == "raise":
            self.range["AA"] *= 1.2
            self.range["KK"] *= 1.1
            self.range["random"] *= 0.8

        elif action == "call":
            self.range["random"] *= 1.1

        self._normalize()

    def _normalize(self):
        total = sum(self.range.values())
        for k in self.range:
            self.range[k] /= total