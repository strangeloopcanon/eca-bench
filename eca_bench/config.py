from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskConfig:
    width: int = 64
    horizon: int = 4
    train_examples: int = 40_000
    val_examples: int = 2_000
    test_examples: int = 10_000
    rules: str | list[int] = "all"
    seed: int = 1337


@dataclass
class ModelConfig:
    pack_bits: int = 1
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: float = 3.0
    dropout: float = 0.0
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 128
    steps: int = 1_200
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 8
    device: str = "auto"


@dataclass
class RunConfig:
    name: str = "baseline"
    output_dir: str = "artifacts"
    target_accuracy: float = 0.99
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_run_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    base = asdict(RunConfig())
    _deep_update(base, raw)
    if overrides:
        _deep_update(base, overrides)

    task = TaskConfig(**base.get("task", {}))
    model = ModelConfig(**base.get("model", {}))
    train = TrainConfig(**base.get("train", {}))

    return RunConfig(
        name=base.get("name", "baseline"),
        output_dir=base.get("output_dir", "artifacts"),
        target_accuracy=float(base.get("target_accuracy", 0.99)),
        task=task,
        model=model,
        train=train,
    )


def save_run_config(config: RunConfig, path: str | Path) -> None:
    payload = asdict(config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
