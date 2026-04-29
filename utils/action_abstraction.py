class ActionAbstraction:
    """
    Reduce action space continuo → discreto manejable
    """

    def __init__(self):
        self.actions = ["fold", "call", "small_raise", "big_raise"]

    def get_actions(self):
        return self.actions

    def normalize(self, action):
        """
        Map raw action → abstract action
        """
        mapping = {
            "fold": "fold",
            "call": "call",
            "raise_1": "small_raise",
            "raise_2": "big_raise",
            "allin": "big_raise"
        }

        return mapping.get(action, "call")