import pytest

from env.poker_env import PokerEnv


def test_heads_up_fold_reward_is_normalized_in_big_blinds():
    env = PokerEnv(num_players=2, starting_stack=10000, small_blind=50, big_blind=100, reward_unit="bb")
    _, reward, done, info = env.step("fold")

    assert done is True
    assert reward == -0.5
    assert info["terminal_utilities"] == [-0.5, 0.5]


def test_six_max_starts_action_left_of_big_blind():
    env = PokerEnv(num_players=6, starting_stack=10000, small_blind=50, big_blind=100, reward_unit="bb")
    state = env.reset()

    assert state["pot"] == 150.0
    assert state["button"] == 1
    assert state["current_player"] == 4


@pytest.mark.parametrize(
    "num_players,expected_sb,expected_bb,expected_first_preflop",
    [
        (2, 1, 0, 1),
        (6, 2, 3, 4),
        (9, 2, 3, 4),
    ],
)
def test_nlhe_blinds_and_first_to_act_match_table_format(num_players, expected_sb, expected_bb, expected_first_preflop):
    env = PokerEnv(num_players=num_players, starting_stack=10000, small_blind=50, big_blind=100, reward_unit="bb")
    state = env.reset()

    sb = state["contributions"].index(50.0)
    bb = state["contributions"].index(100.0)
    assert sb == expected_sb
    assert bb == expected_bb
    assert state["current_player"] == expected_first_preflop


def test_richer_preflop_action_space_is_available():
    env = PokerEnv(num_players=2, starting_stack=10000, small_blind=50, big_blind=100, reward_unit="bb", seed=3)
    legal = env.get_legal_actions()

    assert "bet_125" in legal
    assert "bet_200" in legal
    assert "all_in" in legal


def test_custom_street_buckets_are_exposed_as_legal_actions():
    env = PokerEnv(
        num_players=2,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        reward_unit="bb",
        street_bet_multipliers={"preflop": {"bet_150": 3.0}, "flop": {"bet_33": 0.33}},
        street_raise_multipliers={"preflop": {"bet_150": 2.5}, "flop": {"bet_100": 1.0}},
        seed=3,
    )

    legal = env.get_legal_actions()
    assert "bet_150" in legal


def test_terminal_utilities_are_zero_sum_in_heads_up():
    env = PokerEnv(num_players=2, starting_stack=10000, small_blind=50, big_blind=100, reward_unit="bb")
    _, _, _, info = env.step("fold")

    assert sum(info["terminal_utilities"]) == 0.0


def test_terminal_utilities_stay_zero_sum_in_multiway_foldout():
    env = PokerEnv(num_players=3, starting_stack=10000, small_blind=50, big_blind=100, reward_unit="bb")
    state = env._get_state()
    done = False
    info = {}

    while not done:
        legal = env.get_legal_actions()
        action = "fold" if "fold" in legal else legal[0]
        state, _, done, info = env.step(action)

    assert state["done"] is True
    assert sum(info["terminal_utilities"]) == 0.0


def test_postflop_first_to_act_is_left_of_button_in_multiway():
    env = PokerEnv(num_players=6, starting_stack=10000, small_blind=50, big_blind=100, reward_unit="bb")
    state = env.reset()  # button=1, sb=2, bb=3, utg=4

    # Complete preflop with all calls/checks to reach flop.
    while state["street"] == "preflop" and not state["done"]:
        legal = env.get_legal_actions()
        action = "call" if "call" in legal else ("check" if "check" in legal else legal[0])
        state, _, _, _ = env.step(action)

    assert state["street"] == "flop"
    assert state["current_player"] == 2  # first active seat left of button
