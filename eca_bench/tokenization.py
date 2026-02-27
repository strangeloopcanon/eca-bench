from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PackedTokenizer:
    width: int
    horizon: int
    pack_bits: int = 1

    PAD: int = 0
    BOS: int = 1
    SEP: int = 2
    ROW: int = 3
    EOS: int = 4
    RULE_OFFSET: int = 5

    def __post_init__(self) -> None:
        if self.pack_bits <= 0:
            raise ValueError("pack_bits must be >= 1")
        if self.width % self.pack_bits != 0:
            raise ValueError(f"width={self.width} must be divisible by pack_bits={self.pack_bits}")
        self.state_offset = self.RULE_OFFSET + 256
        self.state_vocab = 1 << self.pack_bits
        self.vocab_size = self.state_offset + self.state_vocab
        self.row_tokens = self.width // self.pack_bits
        self.prefix_length = self.row_tokens + 4
        self.target_length = (self.horizon * self.row_tokens) + (self.horizon - 1) + 1
        self.total_length = self.prefix_length + self.target_length

        self._weights = (2 ** np.arange(self.pack_bits - 1, -1, -1)).astype(np.int64)

    def _pack_row_values(self, row_bits: np.ndarray) -> np.ndarray:
        row_bits = row_bits.astype(np.int64)
        if self.pack_bits == 1:
            return row_bits
        chunks = row_bits.reshape(-1, self.pack_bits)
        return (chunks * self._weights).sum(axis=1)

    def _unpack_row_values(self, packed_vals: np.ndarray) -> np.ndarray:
        packed_vals = packed_vals.astype(np.int64)
        if self.pack_bits == 1:
            return packed_vals.astype(np.uint8)
        bits = ((packed_vals[:, None] >> np.arange(self.pack_bits - 1, -1, -1)) & 1).astype(np.uint8)
        return bits.reshape(-1)

    def encode_row(self, row_bits: np.ndarray) -> np.ndarray:
        vals = self._pack_row_values(row_bits)
        return vals + self.state_offset

    def decode_row(self, row_tokens: np.ndarray) -> np.ndarray:
        vals = row_tokens.astype(np.int64) - self.state_offset
        if vals.min() < 0 or vals.max() >= self.state_vocab:
            raise ValueError("row_tokens out of range for this tokenizer")
        return self._unpack_row_values(vals)

    def encode_sample(
        self,
        rule_id: int,
        init_row: np.ndarray,
        rollout: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rollout.shape != (self.horizon, self.width):
            raise ValueError("rollout shape mismatch")

        prefix = [self.BOS, self.RULE_OFFSET + int(rule_id), self.SEP]
        prefix.extend(self.encode_row(init_row).tolist())
        prefix.append(self.SEP)

        target: list[int] = []
        for t in range(self.horizon):
            target.extend(self.encode_row(rollout[t]).tolist())
            if t < self.horizon - 1:
                target.append(self.ROW)
        target.append(self.EOS)

        full = np.asarray(prefix + target, dtype=np.int64)
        loss_mask = np.zeros_like(full, dtype=np.bool_)
        loss_mask[len(prefix) :] = True

        if len(full) != self.total_length:
            raise AssertionError("encoded sequence length mismatch")
        return full, loss_mask

    def decode_target(self, target_tokens: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        row_tokens: list[int] = []

        for tok in target_tokens.tolist():
            if tok == self.EOS:
                if row_tokens:
                    rows.append(self.decode_row(np.asarray(row_tokens, dtype=np.int64)))
                break
            if tok == self.ROW:
                rows.append(self.decode_row(np.asarray(row_tokens, dtype=np.int64)))
                row_tokens = []
                continue
            row_tokens.append(tok)

        if len(rows) != self.horizon:
            raise ValueError(f"expected {self.horizon} rows, got {len(rows)}")
        return np.stack(rows, axis=0)
