import numpy as np


def compute_advantages(action_values, strategy=None):
    """
    action_values: dict {action: value}
    strategy: optional dict {action: prob}

    In CFR, the baseline is the strategy-weighted state value, not the
    unweighted arithmetic mean across actions.
    """

    if strategy is None:
        avg = np.mean(list(action_values.values()))
    else:
        avg = sum(strategy[a] * action_values[a] for a in action_values)

    return {a: v - avg for a, v in action_values.items()}
