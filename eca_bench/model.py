from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerShape:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    dropout: float = 0.0
    tie_embeddings: bool = True


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.ln2(x)
        h = self.fc2(self.dropout(F.gelu(self.fc1(h))))
        x = x + self.dropout(h)
        return x


class TinyDecoderTransformer(nn.Module):
    def __init__(self, shape: TransformerShape):
        super().__init__()
        if shape.d_model % shape.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if shape.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.shape = shape
        self.token_emb = nn.Embedding(shape.vocab_size, shape.d_model)
        self.pos_emb = nn.Embedding(shape.max_seq_len, shape.d_model)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=shape.d_model,
                    n_heads=shape.n_heads,
                    ffn_dim=shape.ffn_dim,
                    dropout=shape.dropout,
                )
                for _ in range(shape.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(shape.d_model)
        self.lm_head = nn.Linear(shape.d_model, shape.vocab_size, bias=False)

        if shape.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.register_buffer("_causal_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def _get_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask.numel() == 0 or self._causal_mask.shape[0] < length:
            mask = torch.ones((length, length), dtype=torch.bool, device=device).triu(1)
            self._causal_mask = mask
        return self._causal_mask[:length, :length]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.shape.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.shape.max_seq_len}")

        pos = torch.arange(seq_len, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]

        mask = self._get_causal_mask(seq_len, tokens.device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits



def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
