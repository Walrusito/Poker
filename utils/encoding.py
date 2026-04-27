import numpy as np
from typing import Any, Dict

from utils.information_set import InformationSetBuilder


def encode_state(state: Dict[str, Any], player: int = None) -> np.ndarray:
    """
    Encode a poker state into the same engineered feature space used by training.
    """

    if player is None:
        player = state.get("current_player", 0)

    builder = InformationSetBuilder()
    return np.array(builder.encode_vector(state, player), dtype=np.float32)
