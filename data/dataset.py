import torch
from torch.utils.data import Dataset as TorchDataset


class AdvantageDataset(TorchDataset):
    """
    Dataset para RegretNet.  Cada item: (features, action_idx, advantage).

    Optimisation: uses buffer.snapshot() (direct reference, no copy).
    DataLoader(shuffle=True) handles random ordering.
    """

    def __init__(self, buffer):
        self.data = buffer.snapshot()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        x, a_idx, adv = self.data[idx]
        return (
            x.float(),
            torch.tensor(a_idx, dtype=torch.long),
            torch.tensor(adv, dtype=torch.float32),
        )


class PolicyDataset(TorchDataset):
    """
    Dataset para PolicyNet.  Cada item: (features, strategy_vector).
    """

    def __init__(self, buffer):
        self.data = buffer.snapshot()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        x, strategy = self.data[idx]
        return x.float(), torch.tensor(strategy, dtype=torch.float32)
