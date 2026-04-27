from utils.action_abstraction import DEFAULT_ACTIONS


class BetSizingAbstraction:
    """
    Discretizes no-limit bet sizing into a richer action set that depends on street
    and whether the player is opening or raising over an existing bet.
    """

    DEFAULT_STREET_BET_MULTIPLIERS = {
        "preflop": {"bet_125": 2.25, "bet_200": 3.00},
        "flop": {"bet_25": 0.25, "bet_50": 0.50, "bet_75": 0.75, "bet_100": 1.00},
        "turn": {"bet_50": 0.50, "bet_75": 0.75, "bet_100": 1.00, "bet_125": 1.25},
        "river": {"bet_50": 0.50, "bet_75": 0.75, "bet_100": 1.00, "bet_125": 1.25},
    }

    DEFAULT_STREET_RAISE_MULTIPLIERS = {
        "preflop": {"bet_125": 2.20, "bet_200": 3.00},
        "flop": {"bet_75": 0.75, "bet_100": 1.00, "bet_125": 1.25, "bet_200": 2.00},
        "turn": {"bet_75": 0.75, "bet_100": 1.00, "bet_125": 1.25, "bet_200": 2.00},
        "river": {"bet_75": 0.75, "bet_100": 1.00, "bet_125": 1.25, "bet_200": 2.00},
    }

    def __init__(self, street_bet_multipliers=None, street_raise_multipliers=None):
        self.street_bet_multipliers = dict(street_bet_multipliers or self.DEFAULT_STREET_BET_MULTIPLIERS)
        self.street_raise_multipliers = dict(street_raise_multipliers or self.DEFAULT_STREET_RAISE_MULTIPLIERS)
        self.bet_actions = self._build_ordered_action_list()
        self.actions = tuple(["fold", "check", "call"] + self.bet_actions + ["all_in"])

    def _build_ordered_action_list(self):
        labels = set()
        for mapping in self.street_bet_multipliers.values():
            labels.update(mapping.keys())
        for mapping in self.street_raise_multipliers.values():
            labels.update(mapping.keys())

        def _bet_rank(label):
            if not label.startswith("bet_"):
                return 10_000
            try:
                return int(label.split("_", 1)[1])
            except ValueError:
                return 10_000

        return sorted(labels, key=_bet_rank)

    def _street_key(self, street):
        return street if street in self.street_bet_multipliers else "river"

    def _spr(self, pot, stack, to_call, big_blind):
        investable_stack = max(float(stack) - float(to_call), 0.0)
        denominator = max(float(pot), float(big_blind), 1.0)
        return investable_stack / denominator

    def _raise_size(self, action, pot, stack, min_raise, to_call, street, big_blind):
        street_key = self._street_key(street)
        bet_map = self.street_bet_multipliers.get(street_key, {})
        raise_map = self.street_raise_multipliers.get(street_key, {})

        if street_key == "preflop":
            if to_call <= 0:
                target_total = bet_map[action] * float(big_blind)
                return max(float(min_raise), target_total)

            target_raise = raise_map[action] * float(to_call)
            return max(float(min_raise), target_raise)

        if to_call <= 0:
            target_raise = bet_map[action] * max(float(pot), float(big_blind))
            return max(float(min_raise), target_raise)

        target_raise = raise_map[action] * max(float(pot), float(big_blind))
        return max(float(min_raise), target_raise)

    def get_actions(self, pot, stack, min_raise, to_call=0, street="preflop", big_blind=100):
        actions = ["fold", "call"] if to_call > 0 else ["check"]

        if stack <= to_call:
            return actions

        spr = self._spr(pot, stack, to_call, big_blind)
        street_key = self._street_key(street)
        action_candidates = []
        if to_call <= 0:
            action_candidates = [a for a in self.bet_actions if a in self.street_bet_multipliers.get(street_key, {})]
        else:
            action_candidates = [a for a in self.bet_actions if a in self.street_raise_multipliers.get(street_key, {})]

        for action in action_candidates:
            raise_size = self._raise_size(action, pot, stack, min_raise, to_call, street, big_blind)
            total_amount = float(to_call) + raise_size

            if total_amount >= float(stack):
                continue

            if spr < 1.5 and action in {"bet_125", "bet_200"}:
                continue
            if spr < 0.9 and action in {"bet_100", "bet_125", "bet_200"}:
                continue

            actions.append(action)

        actions.append("all_in")
        return actions

    def to_amount(self, action, pot, stack, min_raise, to_call=0, street="preflop", big_blind=100):
        if action == "call":
            return min(float(to_call), float(stack))

        if action in {"check", "fold"}:
            return 0.0

        if action == "all_in":
            return float(stack)

        raise_size = self._raise_size(action, pot, stack, min_raise, to_call, street, big_blind)
        return min(float(stack), float(to_call) + raise_size)
