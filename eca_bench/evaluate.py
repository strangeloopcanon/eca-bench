from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import RunConfig, TaskConfig, load_run_config
from .data import CASequenceDataset, generate_examples
from .model import TinyDecoderTransformer, TransformerShape
from .tokenization import PackedTokenizer
from .train import choose_device, exact_match_accuracy



def _load_shape(run_dir: Path, cfg: RunConfig, tokenizer: PackedTokenizer) -> TransformerShape:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        raw_shape = metrics.get("shape")
        if raw_shape:
            return TransformerShape(**raw_shape)

    return TransformerShape(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.total_length,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        ffn_dim=max(4, int(round(cfg.model.d_model * cfg.model.ffn_mult))),
        dropout=cfg.model.dropout,
        tie_embeddings=cfg.model.tie_embeddings,
    )



def evaluate_run(
    run_dir: str | Path,
    split: str = "test",
    n_examples: int | None = None,
    batch_size: int = 256,
    device: str = "auto",
    task_overrides: TaskConfig | None = None,
) -> dict:
    run_path = Path(run_dir)
    cfg = load_run_config(run_path / "run_config.yaml")

    task = cfg.task
    if task_overrides is not None:
        task = task_overrides

    if n_examples is None:
        if split == "train":
            n_examples = task.train_examples
        elif split == "val":
            n_examples = task.val_examples
        else:
            n_examples = task.test_examples

    tokenizer = PackedTokenizer(width=task.width, horizon=task.horizon, pack_bits=cfg.model.pack_bits)
    shape = _load_shape(run_path, cfg, tokenizer)

    torch_device = choose_device(device)
    model = TinyDecoderTransformer(shape).to(torch_device)
    state = torch.load(run_path / "model.pt", map_location=torch_device)
    model.load_state_dict(state)

    examples = generate_examples(task=task, split=split, n_examples=n_examples)
    dataset = CASequenceDataset(examples, tokenizer)

    exact = exact_match_accuracy(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=torch_device,
        max_examples=n_examples,
    )

    result = {
        "run_dir": str(run_path),
        "split": split,
        "examples": int(n_examples),
        "exact_match": float(exact),
        "target_accuracy": cfg.target_accuracy,
        "target_met": bool(exact >= cfg.target_accuracy),
    }

    out_path = run_path / f"eval_{split}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result
