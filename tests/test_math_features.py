import math

import pytest

from utils.math_features import compute_effective_stack, compute_implied_odds, compute_pot_odds


def test_pot_odds_matches_break_even_threshold():
    assert compute_pot_odds(100.0, 50.0) == pytest.approx(50.0 / 150.0)


def test_implied_odds_uses_future_effective_stack():
    effective = compute_effective_stack(hero_stack=500.0, opponent_stacks=[900.0], to_call=100.0)
    assert effective == pytest.approx(400.0)
    assert compute_implied_odds(200.0, 100.0, effective) == pytest.approx(100.0 / 700.0)


def test_zero_call_cost_has_zero_pot_and_implied_odds():
    assert compute_pot_odds(250.0, 0.0) == 0.0
    assert compute_implied_odds(250.0, 0.0, 400.0) == 0.0


def test_pot_odds_is_clamped_to_unit_interval():
    assert 0.0 <= compute_pot_odds(200.0, 100.0) <= 1.0
    assert compute_pot_odds(-200.0, 100.0) == 1.0


def test_pot_odds_monotonicity():
    base = compute_pot_odds(200.0, 50.0)
    higher_to_call = compute_pot_odds(200.0, 100.0)
    bigger_pot = compute_pot_odds(400.0, 50.0)
    assert higher_to_call > base
    assert bigger_pot < base


def test_pot_odds_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        compute_pot_odds(math.inf, 10.0)
    with pytest.raises(ValueError):
        compute_pot_odds(100.0, math.nan)
