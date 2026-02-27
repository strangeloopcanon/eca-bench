# ECA-Bench: Smallest Transformer for Cellular Automata

Find the smallest transformer that can predict elementary cellular automata.

Inspired by [Dimitris Papailiopoulos's "Addition Under Pressure"](https://x.com/DimitrisPapail/status/1892310069677297932) experiment comparing autonomous coding agents.

## Leaderboard

Ranked by parameter count among submissions with ≥99% sequence-level exact match.

| # | Params | Accuracy | Architecture | Submitter |
|---|--------|----------|--------------|-----------|
| 1 | **448** | 100% | Per-cell features, d_model=8, d_ff=5, 1-layer | claude-4.6-opus |
| 2 | **729** | 100% | Relative attention bias, d_model=8, d_ff=16, 1-layer | gemini-3.1-pro |
| 3 | **56,546** | 100% | Standard transformer, d_model=48, 2-layer | gemini-2.5-pro |

<details>
<summary>All runs (including failures)</summary>

| Submitter | Accuracy | Params | Trained? | Wall-clock | Status |
|-----------|----------|--------|----------|------------|--------|
| claude-4.6-opus | 1.0000 | 448 | Yes | 42 agent turns | PASS |
| gemini-3.1-pro | 1.0000 | 729 | Yes | 54.7 min | PASS |
| gemini-2.5-pro | 1.0000 | 56,546 | Yes | ~2 hr | PASS |
| gpt-5.2 (xhigh) | 0.4089 | 4,336 | Yes | 88.8 min | FAIL |
| gpt-5.2 (xhigh, stalled) | 0.2707 | 3,218 | Yes | ~60 min | FAIL |
| gpt-5.2 | 1.0000 | 0 | No (exploit) | -- | PASS (exploit) |
| gpt-5.3-codex-spark | 1.0000 | 0 | No (exploit) | ~20 min | PASS (exploit) |
| gemini-3-flash | -- | -- | -- | DNF | FAIL |

</details>

## The Challenge

Predict 2 time steps of all 256 one-dimensional elementary cellular automata (width 32, periodic boundary).

| Parameter | Value |
|-----------|-------|
| Rules | All 256 ECA rules (0-255) |
| Width | 32 cells, periodic boundary |
| Horizon | 2 steps |
| Target | ≥99% sequence-level exact match on 10,000 held-out examples |
| Goal | Minimize trainable parameters |

Each ECA step: for each cell, the 3-cell neighborhood (left, center, right) forms a 3-bit index. Output = `(rule >> index) & 1`.

### Constraints

- Must be a **transformer** (contains attention)
- Must be **genuinely trained** via gradient descent (no hardcoded/frozen symbolic solvers)
- No internet, no external tools at inference

## How to Submit

### 1. Set up your workspace

```bash
# Copy the challenge files into a new workspace
mkdir workspaces/your-name
cp open_challenge/{challenge.py,verifier.py} workspaces/your-name/
cp prompt/CHALLENGE.md workspaces/your-name/
```

### 2. Write `submission.py`

Your submission must expose two functions:

```python
import numpy as np

def build_model(seed: int = 0, width: int = 32, horizon: int = 2):
    """Build and return a trained model. Training happens inside this function."""
    ...

def predict_batch(model, rule_ids, init_states, horizon) -> np.ndarray:
    """
    Args:
        model: whatever build_model returned
        rule_ids: np.ndarray shape [batch], int in [0, 255]
        init_states: np.ndarray shape [batch, width], binary {0, 1}
        horizon: int, number of steps to predict
    Returns:
        np.ndarray shape [batch, horizon, width], binary {0, 1}
    """
    ...

# Optional
def count_parameters(model) -> int: ...
METADATA = {"name": "...", "description": "..."}
```

### 3. Score it

```bash
cd workspaces/your-name
python verifier.py --submission submission.py
```

Output:

```json
{"exact_match": 1.0, "target_met": true, "trainable_params": 448, "transformer_like": true}
```

To score all submissions at once:

```bash
./scripts/score.sh
```

## Files

```
open_challenge/
  challenge.py        -- ECA ground truth (read this first)
  verifier.py         -- scorer
prompt/
  CHALLENGE.md        -- full challenge specification
scripts/
  setup.sh            -- create agent workspaces
  score.sh            -- score all submissions
workspaces/*/         -- one folder per submission
```

## Background

This benchmark tests how autonomous coding agents approach an open-ended ML research problem. Eight runs across gpt-5.2, gpt-5.3, claude-4.6-opus, gemini-3.1-pro, gemini-2.5-pro, and gemini-3-flash revealed very different strategies -- from reward hacking to systematic architecture search. See [FINDINGS.md](FINDINGS.md) for the full analysis.
