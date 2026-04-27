from env.rules import evaluate_5, evaluate_7


def card(rank: str, suit: str) -> int:
    ranks = "23456789TJQKA"
    suits = "shdc"
    return suits.index(suit) * 13 + ranks.index(rank)


def test_straight_flush_ranks_above_quads():
    straight_flush = evaluate_5([card("A", "s"), card("K", "s"), card("Q", "s"), card("J", "s"), card("T", "s")])
    quads = evaluate_5([card("A", "s"), card("A", "h"), card("A", "d"), card("A", "c"), card("K", "s")])
    assert straight_flush > quads


def test_wheel_straight_is_detected():
    wheel = evaluate_5([card("A", "s"), card("2", "h"), card("3", "d"), card("4", "c"), card("5", "s")])
    assert wheel == (4, 3)


def test_best_of_seven_selects_strongest_five_cards():
    value = evaluate_7(
        [
            card("A", "s"),
            card("K", "s"),
            card("Q", "s"),
            card("J", "s"),
            card("T", "s"),
            card("2", "h"),
            card("3", "d"),
        ]
    )
    weaker = evaluate_7(
        [
            card("A", "s"),
            card("K", "s"),
            card("Q", "s"),
            card("J", "s"),
            card("9", "s"),
            card("2", "h"),
            card("3", "d"),
        ]
    )
    assert value > weaker
