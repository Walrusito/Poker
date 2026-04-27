import torch
import torch.nn as nn


class RegretNet(nn.Module):
    """
    Maps an information-set feature vector -> per-action regret estimates.
    No final activation: raw values (can be negative).
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, output_dim: int = 5,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
