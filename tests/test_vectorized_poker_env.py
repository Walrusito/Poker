from env.poker_env import PokerEnv
from env.vectorized_poker_env import VectorizedPokerEnv


def test_vectorized_env_step_shapes():
    template = PokerEnv(num_players=2, starting_stack=1000, small_blind=5, big_blind=10, reward_unit="bb", seed=2)
    vec = VectorizedPokerEnv.from_template(template, batch_size=3)
    states = vec.reset()
    assert len(states) == 3

    actions = []
    for env in vec.envs:
        legal = env.get_legal_actions()
        actions.append("call" if "call" in legal else legal[0])
    next_states, rewards, dones, infos = vec.step(actions)

    assert len(next_states) == 3
    assert len(rewards) == 3
    assert len(dones) == 3
    assert len(infos) == 3
