class CFRNode:
    """
    Nodo del árbol CFR (information set node).

    FIX: get_strategy() now guards against an empty regret dict to avoid
    ZeroDivisionError when a node is created but no actions have been
    registered yet.
    """

    def __init__(self, info_set: str):
        self.info_set = info_set
        self.regret: dict = {}
        self.strategy: dict = {}
        self.strategy_sum: dict = {}

    def get_strategy(self, reach_prob: float) -> dict:
        if not self.regret:
            return {}

        normalizer = sum(max(r, 0) for r in self.regret.values())

        if normalizer > 0:
            strategy = {a: max(self.regret[a], 0) / normalizer for a in self.regret}
        else:
            n = len(self.regret)
            strategy = {a: 1.0 / n for a in self.regret}

        # Accumulate strategy sum weighted by reach probability
        for a, prob in strategy.items():
            self.strategy_sum[a] = self.strategy_sum.get(a, 0.0) + reach_prob * prob

        return strategy

    def get_average_strategy(self) -> dict:
        """Return the time-averaged strategy (Nash approximation)."""
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {a: v / total for a, v in self.strategy_sum.items()}
        n = len(self.regret)
        if n == 0:
            return {}
        return {a: 1.0 / n for a in self.regret}
