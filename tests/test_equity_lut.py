from utils.equity_lut import EquityLUT


def card(rank: str, suit: str) -> int:
    ranks = "23456789TJQKA"
    suits = "shdc"
    return suits.index(suit) * 13 + ranks.index(rank)


def test_preflop_lut_persists_entries(tmp_path):
    lut = EquityLUT(lut_dir=tmp_path, mc_simulations=20, lut_simulations=20, seed=11)
    hand = [card("A", "s"), card("K", "s")]

    value = lut.estimate(hand, [], num_players=2)
    key = lut.preflop_key(hand, 2)

    assert key in lut.preflop_table
    assert value == lut.preflop_table[key]
    assert (tmp_path / "preflop_equity_lut.json").exists()


def test_flop_lut_persists_entries(tmp_path):
    lut = EquityLUT(lut_dir=tmp_path, mc_simulations=20, lut_simulations=20, seed=5)
    hand = [card("A", "s"), card("K", "s")]
    flop = [card("Q", "s"), card("J", "d"), card("2", "c")]

    value = lut.estimate(hand, flop, num_players=2)
    key = lut.flop_key(hand, flop, 2)

    assert key in lut.flop_table
    assert value == lut.flop_table[key]
    assert (tmp_path / "flop_equity_lut.json").exists()


def test_turn_and_river_lut_persist_entries(tmp_path):
    lut = EquityLUT(lut_dir=tmp_path, mc_simulations=20, lut_simulations=20, seed=5)
    hand = [card("A", "s"), card("K", "s")]
    turn = [card("Q", "s"), card("J", "d"), card("2", "c"), card("8", "h")]
    river = turn + [card("7", "d")]

    turn_value = lut.estimate(hand, turn, num_players=2)
    river_value = lut.estimate(hand, river, num_players=2)

    turn_key = lut.turn_key(hand, turn, 2)
    river_key = lut.river_key(hand, river, 2)
    assert turn_key in lut.turn_table
    assert river_key in lut.river_table
    assert turn_value == lut.turn_table[turn_key]
    assert river_value == lut.river_table[river_key]
    assert (tmp_path / "turn_equity_lut.json").exists()
    assert (tmp_path / "river_equity_lut.json").exists()


def test_equity_stats_include_per_street_rates(tmp_path):
    lut = EquityLUT(lut_dir=tmp_path, mc_simulations=20, lut_simulations=20, seed=13)
    hand = [card("A", "s"), card("K", "s")]
    flop = [card("Q", "s"), card("J", "d"), card("2", "c")]
    turn = flop + [card("8", "h")]

    lut.estimate(hand, [], num_players=2)
    lut.estimate(hand, flop, num_players=2)
    lut.estimate(hand, turn, num_players=2)
    stats = lut.get_stats()

    assert "lut_hit_rate_preflop" in stats
    assert "lut_hit_rate_flop" in stats
    assert "lut_hit_rate_turn" in stats
    assert "lut_hit_rate_river" in stats
    assert "equity_avg_ms" in stats
    assert stats["equity_avg_ms"] >= 0.0
