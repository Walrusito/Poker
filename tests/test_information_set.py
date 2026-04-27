from utils.information_set import InformationSetBuilder


def card(rank: str, suit: str) -> int:
    ranks = "23456789TJQKA"
    suits = "shdc"
    return suits.index(suit) * 13 + ranks.index(rank)


def test_information_set_contains_engineered_odds_features(tmp_path):
    builder = InformationSetBuilder(mc_simulations=10, lut_simulations=10, lut_dir=tmp_path, seed=3, cache_size=16)
    state = {
        "hands": [[card("A", "s"), card("K", "s")], [card("Q", "h"), card("Q", "d")]],
        "board": [],
        "pot": 300.0,
        "bets": [100.0, 200.0],
        "contributions": [100.0, 200.0],
        "stacks": [9800.0, 9700.0],
        "active": [True, True],
        "street": "preflop",
        "button": 0,
        "pending_players": [0, 1],
        "last_raise_size": 200.0,
        "last_aggressor": 1,
        "street_actions": 1,
        "starting_stack": 10000.0,
        "big_blind": 100.0,
        "num_players": 2,
    }

    features = builder.encode(state, player=0)
    cached_features = builder.encode(state, player=0)

    assert "pot_odds" in features
    assert "implied_odds" in features
    assert "players_to_act_norm" in features
    assert "num_players_norm" in features
    assert "hero_contribution_norm" in features
    assert "spr_norm" in features
    assert "last_raise_norm" in features
    assert "aggressor_distance_norm" in features
    assert "has_aggressor" in features
    assert "facing_bet" in features
    assert "is_button" in features
    assert "is_small_blind" in features
    assert "is_big_blind" in features
    assert "board_paired" in features
    assert "board_monotone" in features
    assert "board_connected" in features
    assert "board_high_card_density" in features
    assert "blocker_ace" in features
    assert "blocker_king" in features
    assert "nut_potential" in features
    assert "aggression_last4_norm" in features
    assert "last_raise_to_pot_ratio" in features
    assert features["pot_odds"] == 100.0 / 400.0
    assert features["implied_odds"] == 100.0 / 10100.0
    assert features["players_to_act_norm"] == 1.0
    assert features["num_players_norm"] == 2.0 / 9.0
    assert features["hero_contribution_norm"] == 0.01
    assert features["last_raise_norm"] == 0.02
    assert features["aggressor_distance_norm"] == 1.0
    assert features["has_aggressor"] == 1.0
    assert features["facing_bet"] == 1.0
    assert features["is_button"] == 1.0
    assert features["is_small_blind"] == 1.0
    assert features["is_big_blind"] == 0.0
    assert 0.0 <= features["spr_norm"] <= 1.0
    assert 0.0 <= features["nut_potential"] <= 1.0
    assert cached_features is features
    assert len(builder._feature_cache) == 1


def test_information_set_blind_flags_follow_multiway_positions(tmp_path):
    builder = InformationSetBuilder(mc_simulations=10, lut_simulations=10, lut_dir=tmp_path, seed=7, cache_size=0)
    state = {
        "hands": [
            [card("A", "s"), card("K", "s")],
            [card("Q", "h"), card("Q", "d")],
            [card("J", "h"), card("T", "h")],
            [card("9", "c"), card("9", "d")],
        ],
        "board": [],
        "pot": 150.0,
        "bets": [0.0, 50.0, 100.0, 0.0],
        "contributions": [0.0, 50.0, 100.0, 0.0],
        "stacks": [10000.0, 9950.0, 9900.0, 10000.0],
        "active": [True, True, True, True],
        "street": "preflop",
        "button": 0,
        "pending_players": [3, 0, 1, 2],
        "last_raise_size": 100.0,
        "last_aggressor": 2,
        "street_actions": 1,
        "starting_stack": 10000.0,
        "big_blind": 100.0,
        "num_players": 4,
    }

    sb_features = builder.encode(state, player=1)
    bb_features = builder.encode(state, player=2)
    button_features = builder.encode(state, player=0)

    assert sb_features["is_small_blind"] == 1.0
    assert sb_features["is_big_blind"] == 0.0
    assert bb_features["is_small_blind"] == 0.0
    assert bb_features["is_big_blind"] == 1.0
    assert button_features["is_button"] == 1.0


def test_information_set_cache_key_distinguishes_button_and_pending_order(tmp_path):
    builder = InformationSetBuilder(mc_simulations=10, lut_simulations=10, lut_dir=tmp_path, seed=19, cache_size=16)
    base_state = {
        "hands": [[card("A", "s"), card("K", "s")], [card("Q", "h"), card("Q", "d")]],
        "board": [],
        "pot": 300.0,
        "bets": [100.0, 200.0],
        "contributions": [100.0, 200.0],
        "stacks": [9800.0, 9700.0],
        "active": [True, True],
        "street": "preflop",
        "button": 0,
        "pending_players": [0, 1],
        "last_raise_size": 200.0,
        "last_aggressor": 1,
        "street_actions": 1,
        "starting_stack": 10000.0,
        "big_blind": 100.0,
        "num_players": 2,
    }

    first_features = builder.encode(base_state, player=0)
    altered_state = dict(base_state)
    altered_state["button"] = 1
    altered_state["pending_players"] = [1, 0]
    second_features = builder.encode(altered_state, player=0)

    assert first_features is not second_features
    assert first_features["is_button"] == 1.0
    assert second_features["is_button"] == 0.0
