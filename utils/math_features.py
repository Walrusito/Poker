import math
from typing import Iterable


def compute_pot_odds(pot: float, to_call: float) -> float:
    if not math.isfinite(pot) or not math.isfinite(to_call):
        raise ValueError("pot and to_call must be finite numbers")
    if to_call <= 0:
        return 0.0
    denominator = pot + to_call
    if denominator <= 0:
        return 1.0
    odds = to_call / denominator
    return max(0.0, min(1.0, odds))


def compute_effective_stack(hero_stack: float, opponent_stacks: Iterable[float], to_call: float) -> float:
    hero_after_call = max(hero_stack - to_call, 0.0)
    live_opponents = [max(stack, 0.0) for stack in opponent_stacks if stack > 0]
    if not live_opponents:
        return 0.0
    return min(hero_after_call, max(live_opponents))


def compute_implied_odds(pot: float, to_call: float, effective_stack: float) -> float:
    if to_call <= 0:
        return 0.0
    return to_call / (pot + to_call + max(effective_stack, 0.0))
