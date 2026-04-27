from collections import OrderedDict
from typing import Mapping

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

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = max(1, int(num_layers))
        self.dropout = float(dropout)

        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    @property
    def _backbone(self):
        return self.net

    @staticmethod
    def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]):
        normalized = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("_backbone."):
                key = "net." + key[len("_backbone."):]
            normalized[key] = value
        return normalized

    @classmethod
    def infer_architecture_from_state_dict(cls, state_dict: Mapping[str, torch.Tensor]):
        normalized = cls.normalize_state_dict_keys(state_dict)
        linear_layers = sorted(
            (
                (int(key.split(".")[1]), value.shape)
                for key, value in normalized.items()
                if key.startswith("net.") and key.endswith(".weight") and getattr(value, "ndim", 0) == 2
            ),
            key=lambda item: item[0],
        )
        if not linear_layers:
            raise ValueError("Could not infer PolicyNet architecture from checkpoint state dict")

        first_shape = linear_layers[0][1]
        last_shape = linear_layers[-1][1]
        return {
            "input_dim": int(first_shape[1]),
            "hidden_dim": int(first_shape[0]),
            "output_dim": int(last_shape[0]),
            "num_layers": max(1, len(linear_layers) - 1),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload, input_dim=None, output_dim=None, device=None):
        policy_state = payload["policy_net_state"]
        inferred = cls.infer_architecture_from_state_dict(policy_state)
        config = dict(payload.get("config") or {})

        checkpoint_input_dim = int(config.get("input_dim", inferred["input_dim"]))
        checkpoint_output_dim = int(config.get("output_dim", inferred["output_dim"]))
        resolved_input_dim = checkpoint_input_dim if input_dim is None else int(input_dim)
        resolved_output_dim = checkpoint_output_dim if output_dim is None else int(output_dim)

        if resolved_input_dim != checkpoint_input_dim or resolved_output_dim != checkpoint_output_dim:
            raise ValueError(
                "Checkpoint PolicyNet dimensions do not match the requested feature/action space. "
                f"checkpoint=({checkpoint_input_dim}, {checkpoint_output_dim}) "
                f"requested=({resolved_input_dim}, {resolved_output_dim})"
            )

        model = cls(
            input_dim=resolved_input_dim,
            hidden_dim=int(config.get("hidden_dim", inferred["hidden_dim"])),
            output_dim=resolved_output_dim,
            num_layers=int(config.get("num_layers", inferred["num_layers"])),
            dropout=float(config.get("dropout", 0.0)),
        )
        if device is not None:
            model = model.to(device)
        model.load_state_dict(policy_state)
        return model

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        normalized = self.normalize_state_dict_keys(state_dict)
        try:
            return super().load_state_dict(normalized, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(normalized, strict=strict)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)
