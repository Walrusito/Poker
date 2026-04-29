import random
from typing import List, Tuple, Any


def self_play(env, episodes: int = 10, policy=None) -> List[Tuple]:
    """
    Collect (state, action, reward) tuples by running episodes.

    FIX: original always played "call" regardless of state.
    Now supports an optional policy callable:
      policy(state) -> action_str
    If no policy is provided, actions are chosen uniformly at random
    from the legal actions — still random but at least varies.
    """
    data = []

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            legal = env.get_legal_actions()

            if policy is not None:
                action = policy(state)
                # Fallback to random if policy returns an illegal action
                if action not in legal:
                    action = random.choice(legal)
            else:
                action = random.choice(legal)

            state, reward, done, _ = env.step(action)
            data.append((state, action, reward))

    return data
