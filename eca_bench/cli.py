from __future__ import annotations

import argparse
import json

from .config import load_run_config
from .evaluate import evaluate_run
from .train import train_once


def train_cli() -> None:
    parser = argparse.ArgumentParser(description="Train one CA transformer candidate")
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    summary = train_once(cfg)
    print(json.dumps(summary, indent=2))


def evaluate_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run directory")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = evaluate_run(
        run_dir=args.run_dir,
        split=args.split,
        n_examples=args.examples,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
