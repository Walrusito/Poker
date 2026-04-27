import copy
from typing import Dict, List, Sequence, Tuple

from env.poker_env import PokerEnv


class VectorizedPokerEnv:
    """
    Batched wrapper around PokerEnv instances.

    This is an orchestration layer (many envs in lockstep-friendly API), not a full
    tensor-native poker engine. It enables batched policy inference and rollout flow.
    """

    def __init__(self, envs: Sequence[PokerEnv]):
        self.envs = list(envs)

    @classmethod
    def from_template(cls, template_env: PokerEnv, batch_size: int):
        envs = []
        for _ in range(max(1, int(batch_size))):
            envs.append(copy.deepcopy(template_env))
        return cls(envs)

    def reset(self) -> List[Dict]:
        states = []
        for env in self.envs:
            states.append(env.reset())
        return states

    def get_states(self) -> List[Dict]:
        return [env._get_state() for env in self.envs]

    def active_indices(self) -> List[int]:
        return [idx for idx, env in enumerate(self.envs) if not env.done]

    def step(self, actions: Sequence[str]) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        if len(actions) != len(self.envs):
            raise ValueError(f"Expected {len(self.envs)} actions, got {len(actions)}")

        states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            if env.done:
                state = env._get_state()
                states.append(state)
                rewards.append(0.0)
                dones.append(True)
                infos.append({"terminal_utilities": env.get_terminal_utilities()})
                continue
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return states, rewards, dones, infos
