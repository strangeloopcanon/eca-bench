from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import RunConfig, save_run_config
from .data import CASequenceDataset, generate_examples
from .model import TinyDecoderTransformer, TransformerShape, trainable_parameter_count
from .tokenization import PackedTokenizer


def choose_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_next_token_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    vocab = logits.shape[-1]
    pred = logits[:, :-1, :].reshape(-1, vocab)
    target = tokens[:, 1:].reshape(-1)
    mask = loss_mask[:, 1:].reshape(-1).float()

    losses = F.cross_entropy(pred, target, reduction="none")
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def generate_targets(
    model: TinyDecoderTransformer,
    prefix_tokens: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    generated = prefix_tokens
    for _ in range(target_len):
        logits = model(generated)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
    return generated[:, prefix_tokens.shape[1] :]


@torch.no_grad()
def exact_match_accuracy(
    model: TinyDecoderTransformer,
    dataset: CASequenceDataset,
    batch_size: int,
    device: torch.device,
    max_examples: int | None = None,
) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    tokenizer = dataset.tokenizer
    prefix_len = tokenizer.prefix_length
    target_len = tokenizer.target_length

    total = 0
    correct = 0

    for batch in loader:
        tokens = batch["tokens"].to(device)
        if max_examples is not None and total >= max_examples:
            break
        if max_examples is not None and total + tokens.shape[0] > max_examples:
            tokens = tokens[: max_examples - total]

        prefix = tokens[:, :prefix_len]
        target = tokens[:, prefix_len : prefix_len + target_len]

        pred = generate_targets(model, prefix, target_len)
        match = (pred == target).all(dim=1)

        correct += int(match.sum().item())
        total += tokens.shape[0]

    if total == 0:
        return 0.0
    return correct / total



def _run_dir(base_dir: str | Path, name: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return Path(base_dir) / f"{name}-{stamp}"



def train_once(config: RunConfig) -> dict:
    set_seed(config.task.seed)

    tokenizer = PackedTokenizer(
        width=config.task.width,
        horizon=config.task.horizon,
        pack_bits=config.model.pack_bits,
    )

    train_examples = generate_examples(config.task, split="train", n_examples=config.task.train_examples)
    val_examples = generate_examples(config.task, split="val", n_examples=config.task.val_examples)

    train_ds = CASequenceDataset(train_examples, tokenizer)
    val_ds = CASequenceDataset(val_examples, tokenizer)

    shape = TransformerShape(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.total_length,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        ffn_dim=max(4, int(round(config.model.d_model * config.model.ffn_mult))),
        dropout=config.model.dropout,
        tie_embeddings=config.model.tie_embeddings,
    )

    device = choose_device(config.train.device)
    model = TinyDecoderTransformer(shape).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    history: list[dict] = []
    best_val_exact = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    last_loss = None

    for step in range(1, config.train.steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens = batch["tokens"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        lr = config.train.learning_rate
        if config.train.warmup_steps > 0 and step <= config.train.warmup_steps:
            lr = config.train.learning_rate * (step / config.train.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = masked_next_token_loss(logits, tokens, loss_mask)
        loss.backward()

        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        optimizer.step()
        last_loss = float(loss.item())

        if step % config.train.eval_every == 0 or step == config.train.steps:
            max_eval = min(config.task.val_examples, config.train.eval_batches * config.train.batch_size)
            val_exact = exact_match_accuracy(
                model=model,
                dataset=val_ds,
                batch_size=config.train.batch_size,
                device=device,
                max_examples=max_eval,
            )
            record = {
                "step": step,
                "train_loss": last_loss,
                "val_exact": val_exact,
                "lr": lr,
            }
            history.append(record)

            if val_exact > best_val_exact:
                best_val_exact = val_exact
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    params = trainable_parameter_count(model)

    run_dir = _run_dir(config.output_dir, config.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(config, run_dir / "run_config.yaml")
    torch.save(model.state_dict(), run_dir / "model.pt")

    metrics = {
        "params": int(params),
        "best_val_exact": float(best_val_exact),
        "last_train_loss": float(last_loss if last_loss is not None else 0.0),
        "history": history,
        "shape": asdict(shape),
        "tokenizer": {
            "width": tokenizer.width,
            "horizon": tokenizer.horizon,
            "pack_bits": tokenizer.pack_bits,
            "vocab_size": tokenizer.vocab_size,
            "total_length": tokenizer.total_length,
            "prefix_length": tokenizer.prefix_length,
            "target_length": tokenizer.target_length,
        },
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = {
        "run_dir": str(run_dir),
        "params": int(params),
        "best_val_exact": float(best_val_exact),
        "target_met": bool(best_val_exact >= config.target_accuracy),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
