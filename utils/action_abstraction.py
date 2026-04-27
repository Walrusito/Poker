DEFAULT_ACTIONS = (
    "fold",
    "check",
    "call",
    "bet_25",
    "bet_50",
    "bet_75",
    "bet_100",
    "bet_125",
    "bet_200",
    "all_in",
)


class ActionAbstraction:
    """
    Maps continuous or raw action labels onto the discrete environment action set.
    """

    def __init__(self, actions=None):
        self.actions = tuple(actions or DEFAULT_ACTIONS)

    def get_actions(self):
        return list(self.actions)

    def normalize(self, action):
        if action in self.actions:
            return action

        mapping = {
            "fold": "fold",
            "check": "check",
            "call": "call",
            "raise_0.25": "bet_25",
            "raise_0.33": "bet_25",
            "raise_0.5": "bet_50",
            "raise_0.75": "bet_75",
            "raise_1.0": "bet_100",
            "raise_1.25": "bet_125",
            "raise_1.5": "bet_125",
            "raise_2.0": "bet_200",
            "all_in": "all_in",
            "jam": "all_in",
            # Legacy aliases kept for compatibility with old agents/checkpoints.
            "bet_small": "bet_50",
            "bet_medium": "bet_75",
            "bet_big": "bet_125",
        }
        return mapping.get(action, "call")
