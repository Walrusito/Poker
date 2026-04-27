import torch
from torch.utils.data import Dataset as TorchDataset


class AdvantageDataset(TorchDataset):
    """
    PyTorch-compatible Dataset wrapping a ReservoirBuffer for the regret net.

    Each item is (features: Tensor, action_idx: int, advantage: float).
    FIX: original Dataset.__getitem__ returned raw tuples without any tensor
    conversion, causing runtime errors when used with DataLoader.
    """

    def __init__(self, buffer):
        self.data = buffer.sample()  # shuffled snapshot

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        x, a_idx, adv = self.data[idx]
        return x.float(), torch.tensor(a_idx, dtype=torch.long), torch.tensor(adv, dtype=torch.float32)


class PolicyDataset(TorchDataset):
    """
    PyTorch-compatible Dataset wrapping a ReservoirBuffer for the policy net.

    Each item is (features: Tensor, strategy: Tensor).
    """

    def __init__(self, buffer):
        self.data = buffer.sample()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        x, strategy = self.data[idx]
        return x.float(), torch.tensor(strategy, dtype=torch.float32)
