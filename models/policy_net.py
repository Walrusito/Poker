import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    Maps an information-set feature vector -> action probability distribution.

    The network outputs raw logits. Use ``forward()`` for probabilities
    (softmax) or ``log_probs()`` for numerically-stable log-probabilities
    (log_softmax) during training.
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
        self._backbone = nn.Sequential(*layers)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self._backbone(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self._backbone(x), dim=-1)
