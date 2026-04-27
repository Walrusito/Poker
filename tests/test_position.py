from utils.position import Position


def test_heads_up_blinds_keep_button_as_small_blind():
    pos = Position(num_players=2)
    pos.button = 1
    sb, bb = pos.blind_positions()
    assert sb == 1
    assert bb == 0


def test_multiway_blinds_place_small_blind_left_of_button():
    pos = Position(num_players=6)
    pos.button = 3
    sb, bb = pos.blind_positions()
    assert sb == 4
    assert bb == 5
