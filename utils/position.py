class Position:
    """
    Positional helper for 2-9 handed simulations.
    """

    __slots__ = ("num_players", "button")

    def __init__(self, num_players: int):
        if not 2 <= num_players <= 9:
            raise ValueError("num_players must be between 2 and 9")

        self.num_players = num_players
        self.button = 0

    def switch_button(self):
        self.button = (self.button + 1) % self.num_players

    def next_player(self, player: int, active=None) -> int:
        for offset in range(1, self.num_players + 1):
            seat = (player + offset) % self.num_players
            if active is None or active[seat]:
                return seat
        raise ValueError("No active player found")

    def blind_positions(self, active=None):
        if self.num_players == 2:
            sb = self.button
        else:
            sb = self.next_player(self.button, active)
        bb = self.next_player(sb, active)
        return sb, bb

    def first_to_act_preflop(self, active=None) -> int:
        _, bb = self.blind_positions(active)
        if self.num_players == 2:
            return self.button
        return self.next_player(bb, active)

    def first_to_act_postflop(self, active=None) -> int:
        return self.next_player(self.button, active)

    def clone(self):
        new = object.__new__(Position)
        new.num_players = self.num_players
        new.button = self.button
        return new

    def relative_position(self, player: int) -> float:
        if self.num_players == 1:
            return 0.0
        distance = (player - self.button) % self.num_players
        return distance / (self.num_players - 1)
