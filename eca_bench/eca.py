from __future__ import annotations

import numpy as np


def step_batch(rule_ids: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Vectorized one-step evolution for elementary cellular automata.

    Args:
        rule_ids: shape [batch], int in [0, 255]
        states: shape [batch, width], binary {0, 1}
    """
    left = np.roll(states, 1, axis=1)
    center = states
    right = np.roll(states, -1, axis=1)
    codes = (left << 2) | (center << 1) | right
    return ((rule_ids[:, None] >> codes) & 1).astype(np.uint8)


def rollout_batch(rule_ids: np.ndarray, initial_states: np.ndarray, horizon: int) -> np.ndarray:
    """Roll out CA trajectories for each sample.

    Returns:
        shape [batch, horizon, width]
    """
    batch_size, width = initial_states.shape
    out = np.zeros((batch_size, horizon, width), dtype=np.uint8)
    current = initial_states.astype(np.uint8, copy=True)
    for t in range(horizon):
        current = step_batch(rule_ids, current)
        out[:, t] = current
    return out
