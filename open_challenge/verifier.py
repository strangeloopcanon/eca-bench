from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from challenge import ChallengeConfig, generate_eval_data



def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def _call_with_supported_kwargs(fn, kwargs: dict[str, Any]):
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**accepted)



def _is_transformer_like(model: Any) -> bool:
    # KNOWN EXPLOIT: This only checks for the *presence* of attention modules,
    # not that they participate in learned computation. An agent can wrap a
    # symbolic CA engine in nn.MultiheadAttention with handcrafted/frozen weights
    # and pass this check while doing zero learning. The gpt-5.2 (Codex) run
    # demonstrated this exploit in Feb 2026.
    #
    # To harden: verify that attention weights have non-trivial distributions,
    # require nonzero requires_grad parameters, or check gradient flow during
    # a probe forward pass.
    try:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            for m in model.modules():
                name = m.__class__.__name__.lower()
                if isinstance(m, nn.MultiheadAttention):
                    return True
                if "transformer" in name or "attention" in name:
                    return True
    except Exception:
        pass

    # Fallback for non-nn.Module wrappers.
    text = str(type(model)).lower()
    return ("transformer" in text) or ("attention" in text)



def _count_params(model: Any, module: Any) -> int:
    # KNOWN EXPLOIT: The submission can override count_parameters to return 0,
    # or freeze all parameters (requires_grad=False) so the fallback count is 0.
    # The gpt-5.2 (Codex) run used both techniques: frozen MHA weights + a
    # count_parameters that hardcodes 0.
    #
    # To harden: ignore the submission's count_parameters override and always
    # use p.numel() over ALL parameters (not just requires_grad=True), or at
    # minimum flag when trainable == 0 as suspicious.
    custom = getattr(module, "count_parameters", None)
    if callable(custom):
        return int(custom(model))

    if hasattr(model, "parameters"):
        try:
            return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        except Exception:
            return 0
    return 0



def _predict(
    module: Any,
    model: Any,
    rules: np.ndarray,
    init_states: np.ndarray,
    horizon: int,
) -> np.ndarray:
    predict_batch = getattr(module, "predict_batch", None)
    if callable(predict_batch):
        out = predict_batch(model, rules, init_states, horizon)
        arr = np.asarray(out, dtype=np.uint8)
        return arr

    predict_one = getattr(module, "predict_one", None)
    if callable(predict_one):
        rows = []
        for rule, init in zip(rules.tolist(), init_states):
            pred = predict_one(model, int(rule), np.asarray(init, dtype=np.uint8), horizon)
            rows.append(np.asarray(pred, dtype=np.uint8))
        return np.stack(rows, axis=0)

    raise RuntimeError("submission must define predict_batch or predict_one")



def verify_submission(
    submission_path: Path,
    config: ChallengeConfig,
    require_transformer: bool = True,
) -> dict[str, Any]:
    start = time.time()

    module = _load_module(submission_path)
    build_model = getattr(module, "build_model", None)
    if not callable(build_model):
        raise RuntimeError("submission must define build_model")

    model = _call_with_supported_kwargs(
        build_model,
        {
            "seed": config.seed,
            "width": config.width,
            "horizon": config.horizon,
        },
    )

    transformer_like = _is_transformer_like(model)
    if require_transformer and not transformer_like:
        raise RuntimeError("Model does not appear transformer-like (missing attention/transformer modules)")

    params = _count_params(model, module)

    rules, init_states, targets = generate_eval_data(config=config, split="test")
    preds = _predict(module, model, rules=rules, init_states=init_states, horizon=config.horizon)

    expected_shape = targets.shape
    if preds.shape != expected_shape:
        raise RuntimeError(f"prediction shape mismatch: expected {expected_shape}, got {preds.shape}")

    exact = (preds == targets).all(axis=(1, 2)).mean().item()

    out = {
        "submission": str(submission_path),
        "config": asdict(config),
        "exact_match": float(exact),
        "target": 0.99,
        "target_met": bool(exact >= 0.99),
        "trainable_params": int(params),
        "transformer_like": bool(transformer_like),
        "metadata": getattr(module, "METADATA", {}),
        "elapsed_sec": round(time.time() - start, 3),
    }
    return out



def main() -> int:
    parser = argparse.ArgumentParser(description="Verify open CA challenge submission")
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--test-examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--no-require-transformer", action="store_true")
    args = parser.parse_args()

    config = ChallengeConfig(
        width=args.width,
        horizon=args.horizon,
        test_examples=args.test_examples,
        seed=args.seed,
    )

    try:
        report = verify_submission(
            submission_path=Path(args.submission),
            config=config,
            require_transformer=not args.no_require_transformer,
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "submission": str(args.submission),
            "error": str(exc),
            "target": 0.99,
            "target_met": False,
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)

    return 0 if report.get("target_met", False) else 1


if __name__ == "__main__":
    sys.exit(main())
