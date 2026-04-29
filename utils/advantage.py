import numpy as np


def compute_advantages(action_values):
    """
    action_values: dict {action: value}

    returns: dict {action: advantage}
    """

    avg = np.mean(list(action_values.values()))

    return {
        a: v - avg
        for a, v in action_values.items()
    }