from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TaskConfig
from .eca import rollout_batch
from .tokenization import PackedTokenizer


def _seed_for_split(seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def resolve_rules(rules: str | list[int]) -> np.ndarray:
    if isinstance(rules, str):
        if rules != "all":
            raise ValueError("rules must be 'all' or list[int]")
        return np.arange(256, dtype=np.int64)

    arr = np.asarray(rules, dtype=np.int64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("rules list must be non-empty")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError("rules must be in [0, 255]")
    return arr


@dataclass
class CAExamples:
    rule_ids: np.ndarray
    init_states: np.ndarray
    rollouts: np.ndarray


def generate_examples(task: TaskConfig, split: str, n_examples: int) -> CAExamples:
    rng = np.random.default_rng(_seed_for_split(task.seed, split))
    allowed_rules = resolve_rules(task.rules)

    rule_ids = rng.choice(allowed_rules, size=n_examples, replace=True).astype(np.int64)
    init_states = rng.integers(0, 2, size=(n_examples, task.width), dtype=np.uint8)
    rollouts = rollout_batch(rule_ids=rule_ids, initial_states=init_states, horizon=task.horizon)

    return CAExamples(rule_ids=rule_ids, init_states=init_states, rollouts=rollouts)


class CASequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: CAExamples, tokenizer: PackedTokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self._tokens = np.zeros((len(examples.rule_ids), tokenizer.total_length), dtype=np.int64)
        self._loss_masks = np.zeros_like(self._tokens, dtype=np.bool_)

        for i in range(len(examples.rule_ids)):
            toks, mask = tokenizer.encode_sample(
                rule_id=int(examples.rule_ids[i]),
                init_row=examples.init_states[i],
                rollout=examples.rollouts[i],
            )
            self._tokens[i] = toks
            self._loss_masks[i] = mask

    def __len__(self) -> int:
        return self._tokens.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.from_numpy(self._tokens[idx]).long(),
            "loss_mask": torch.from_numpy(self._loss_masks[idx]).bool(),
        }
