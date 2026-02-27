from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class ChallengeConfig:
    width: int = 32
    horizon: int = 2
    test_examples: int = 10_000
    seed: int = 1337



def _split_seed(seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF



def eca_step(rule_ids: np.ndarray, states: np.ndarray) -> np.ndarray:
    left = np.roll(states, 1, axis=1)
    center = states
    right = np.roll(states, -1, axis=1)
    neighborhood = (left << 2) | (center << 1) | right
    return ((rule_ids[:, None] >> neighborhood) & 1).astype(np.uint8)



def eca_rollout(rule_ids: np.ndarray, init_states: np.ndarray, horizon: int) -> np.ndarray:
    out = np.zeros((len(rule_ids), horizon, init_states.shape[1]), dtype=np.uint8)
    current = init_states.astype(np.uint8, copy=True)
    for t in range(horizon):
        current = eca_step(rule_ids, current)
        out[:, t] = current
    return out



def generate_eval_data(config: ChallengeConfig, split: str = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_split_seed(config.seed, split))
    rules = rng.integers(0, 256, size=config.test_examples, dtype=np.int64)
    init_states = rng.integers(0, 2, size=(config.test_examples, config.width), dtype=np.uint8)
    targets = eca_rollout(rules, init_states, horizon=config.horizon)
    return rules, init_states, targets
