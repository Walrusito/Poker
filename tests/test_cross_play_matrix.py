import torch

from cross_play_matrix import load_policy
from models.policy_net import PolicyNet


def test_cross_play_load_policy_uses_checkpoint_architecture_and_legacy_keys(tmp_path):
    reference_model = PolicyNet(input_dim=31, hidden_dim=96, output_dim=7, num_layers=3, dropout=0.2)
    legacy_state = {}
    for key, value in reference_model.state_dict().items():
        legacy_key = key.replace("net.", "_backbone.", 1) if key.startswith("net.") else key
        legacy_state[legacy_key] = value.clone()

    checkpoint_path = tmp_path / "policy.pt"
    torch.save(
        {
            "policy_net_state": legacy_state,
            "config": {
                "input_dim": 31,
                "output_dim": 7,
                "hidden_dim": 96,
                "num_layers": 3,
                "dropout": 0.2,
            },
        },
        checkpoint_path,
    )

    loaded = load_policy(checkpoint_path, input_dim=31, output_dim=7, device=torch.device("cpu"))

    assert loaded.hidden_dim == 96
    assert loaded.num_layers == 3
    assert abs(loaded.dropout - 0.2) < 1e-9

