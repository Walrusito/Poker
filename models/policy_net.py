"""
PolicyNet — mapea features del information set → distribución de probabilidad.

OPTIMIZACIONES vs versión original:
  - 3 capas ocultas (mayor profundidad para capturar relaciones complejas)
  - 256 neuronas ocultas
  - LayerNorm para estabilidad de entrenamiento
  - Dropout leve para regularización
  - Inicialización Kaiming
"""

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    Input:  feature vector del information set  [B, input_dim]
    Output: distribución sobre acciones          [B, output_dim]  (suma = 1)
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
            nn.Softmax(dim=-1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Última capa lineal con init pequeño para que Softmax empiece ~uniforme
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last_linear.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
