from utils.hand_equity import HandEquity


def card(rank: str, suit: str) -> int:
    ranks = "23456789TJQKA"
    suits = "shdc"
    return suits.index(suit) * 13 + ranks.index(rank)


def test_exact_river_heads_up_equity_can_be_certain_win():
    estimator = HandEquity(simulations=10, seed=7)
    hand = [card("A", "s"), card("A", "h")]
    board = [card("A", "d"), card("A", "c"), card("2", "s"), card("3", "h"), card("4", "d")]
    assert estimator.estimate(hand, board, num_players=2) == 1.0


def test_exact_river_heads_up_equity_detects_full_board_tie():
    estimator = HandEquity(simulations=10, seed=7)
    hand = [card("2", "c"), card("3", "d")]
    board = [card("A", "s"), card("K", "s"), card("Q", "s"), card("J", "s"), card("T", "s")]
    assert estimator.estimate(hand, board, num_players=2) == 0.5


def test_preflop_equity_drops_when_field_gets_larger():
    estimator = HandEquity(simulations=300, seed=11)
    hand = [card("A", "s"), card("A", "h")]

    heads_up_equity = estimator.estimate(hand, [], num_players=2)
    four_way_equity = estimator.estimate(hand, [], num_players=4)

    assert heads_up_equity > four_way_equity


def test_torch_backend_equity_returns_valid_probability():
    estimator = HandEquity(simulations=30, seed=11, use_torch_backend=True, torch_device="cuda")
    hand = [card("A", "s"), card("A", "h")]
    value = estimator.estimate(hand, [], num_players=3)
    assert 0.0 <= value <= 1.0
