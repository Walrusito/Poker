class Position:
    """
    Handles poker positions in 2-player simplified model
    """

    POSITIONS = ["SB", "BB"]

    def __init__(self):
        self.button = 0  # player 0 starts as SB

    def get_position(self, player):
        if player == self.button:
            return "SB"
        return "BB"

    def switch_button(self):
        self.button = 1 - self.button