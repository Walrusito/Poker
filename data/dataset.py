import torch
from torch.utils.data import Dataset as TorchDataset


class AdvantageDataset(TorchDataset):
    """
    PyTorch-compatible Dataset wrapping a ReservoirBuffer for the regret net.

    Pre-converts all buffer data to stacked tensors in __init__ for
    zero-copy indexing in __getitem__.
    """

    def __init__(self, buffer):
        data = buffer.sample()
        features = []
        actions = []
        advantages = []
        for x, a_idx, adv in data:
            features.append(x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32))
            actions.append(a_idx)
            advantages.append(adv)
        self._features = torch.stack(features, dim=0)
        self._actions = torch.tensor(actions, dtype=torch.long)
        self._advantages = torch.tensor(advantages, dtype=torch.float32)

    def __len__(self) -> int:
        return self._features.size(0)

    def __getitem__(self, idx):
        return self._features[idx], self._actions[idx], self._advantages[idx]


class PolicyDataset(TorchDataset):
    """
    PyTorch-compatible Dataset wrapping a ReservoirBuffer for the policy net.

    Pre-converts all buffer data to stacked tensors in __init__ for
    zero-copy indexing in __getitem__.
    """

    def __init__(self, buffer):
        data = buffer.sample()
        features = []
        strategies = []
        for x, strategy in data:
            features.append(x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32))
            strategies.append(torch.tensor(strategy, dtype=torch.float32) if not isinstance(strategy, torch.Tensor) else strategy.float())
        self._features = torch.stack(features, dim=0)
        self._strategies = torch.stack(strategies, dim=0)

    def __len__(self) -> int:
        return self._features.size(0)

    def __getitem__(self, idx):
        return self._features[idx], self._strategies[idx]
