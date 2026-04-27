import torch
import torch.nn as nn


class RegretNet(nn.Module):
    """
    Maps an information-set feature vector → per-action regret estimates.
    No final activation: raw values (can be negative).

    FIX: input_dim and output_dim are now constructor parameters instead of
    being hardcoded to 3.  This prevents silent shape mismatches when the
    state encoding changes.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, output_dim: int = 5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
