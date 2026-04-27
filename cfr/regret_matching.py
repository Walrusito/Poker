import numpy as np
from typing import Dict


def regret_matching(regrets: Dict[str, float]) -> Dict[str, float]:
    """
    Convierte regrets en estrategia probabilística.
    CFR core idea: positive regrets only.
    """
    positive = {a: max(v, 0.0) for a, v in regrets.items()}
    total = sum(positive.values())

    if total == 0:
        n = len(regrets)
        return {a: 1.0 / n for a in regrets}

    return {a: v / total for a, v in positive.items()}