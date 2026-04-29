"""
RegretNet — estima regrets (ventajas) por acción desde el information set.

OPTIMIZACIONES vs versión original:
  - 3 capas ocultas en lugar de 2 (mayor capacidad de abstracción)
  - 256 neuronas en lugar de 128 (mejor representación de hand buckets)
  - LayerNorm después de cada capa (más estable que BatchNorm en CFR)
  - Sin activación final: los regrets pueden ser negativos ✓
  - Inicialización Kaiming (mejor gradiente inicial con ReLU)
"""

import torch
import torch.nn as nn


class RegretNet(nn.Module):
    """
    Input: feature vector del information set  [B, input_dim]
    Output: regret estimado por acción          [B, output_dim]
    Sin activación final — valores pueden ser negativos.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256,
                 output_dim: int = 3, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
