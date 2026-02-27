# Smallest Transformer for Cellular Automata

You are an autonomous research engineer. Your goal is to find the **smallest possible transformer** that can predict 1D elementary cellular automata (ECA) rollouts with high accuracy.

## Objectives (strict priority order)

1. **Accuracy**: Reach at least **99% sequence-level exact-match** on a held-out test set of 10,000 examples.
2. **Minimize parameters**: Among models that pass the accuracy threshold, find the one with the fewest trainable parameters.
3. **Report**: Produce a `REPORT.md` documenting what you tried, what failed, what succeeded, and your reasoning throughout.

## Task definition

- **Family**: Elementary cellular automata (ECA). Binary, 1D, periodic boundary conditions.
- **Rules**: All 256 rules (rule IDs 0 through 255).
- **Input**: A `rule_id` (integer 0-255) and an `initial_row` (binary array of width 32).
- **Output**: The next 2 rows of the CA evolution, generated **autoregressively** (each row depends on the previous).
- **Width**: 32 cells.
- **Horizon**: 2 time steps.

Each ECA step works as follows: for each cell, look at the 3-cell neighborhood (left, center, right) with periodic boundary. The neighborhood forms a 3-bit number (0-7). The output bit is `(rule_id >> neighborhood) & 1`.

## Hard constraints

- Your model **must be a transformer** (must contain attention mechanism(s)).
- Inference must be **autoregressive generation** of output tokens.
- The model must generalize to a **held-out test set** (not just memorize training data).
- You **cannot** encode the answer in the input representation.
- You **cannot** use a calculator, symbolic solver, or non-neural CA engine at inference time.
- You **cannot** use the internet or any external resources.
- Work **fully autonomously**. Do not ask for clarification. Just come back with the final result.

## Allowed freedom

- You may choose **any data format and tokenization** you want, as long as it is purely programmatic. Pack bits, merge tokens, reverse sequences, invent novel encodings -- whatever you think will help.
- You may choose any **training strategy** (curriculum, data augmentation, learning rate schedule, etc.).
- You may choose any **architecture variant** within the transformer family (decoder-only, encoder-decoder, number of layers, attention heads, FFN width, etc.).
- You may run as many experiments as you need.

## Workspace

Your workspace contains two files you should read first:

- `challenge.py` -- The ground-truth ECA implementation and test data generator. Study this carefully.
- `verifier.py` -- The scoring tool. It will evaluate your submission on a held-out test set.

## Submission interface

Create a file called `submission.py` that exposes:

```python
def build_model(seed: int = 0, width: int = 32, horizon: int = 2):
    """Build and return the trained model. Training can happen inside this function."""
    ...

def predict_batch(model, rule_ids, init_states, horizon) -> np.ndarray:
    """
    Args:
        model: whatever build_model returned
        rule_ids: np.ndarray of shape [batch], int in [0, 255]
        init_states: np.ndarray of shape [batch, width], binary {0,1}
        horizon: int, number of steps to predict
    Returns:
        np.ndarray of shape [batch, horizon, width], binary {0,1}
    """
    ...
```

Optionally also expose:
- `count_parameters(model) -> int`
- `METADATA = {"name": "...", "description": "..."}`

## Verification

Run this to check your submission:

```bash
python verifier.py --submission submission.py --test-examples 10000
```

The verifier checks:
- That your model contains a transformer/attention module
- Sequence-level exact match on the held-out test set
- Trainable parameter count

## What success looks like

- `submission.py` that passes the verifier at >=99% exact match
- The smallest parameter count you can achieve
- `REPORT.md` covering your experiments, failures, and final approach

## Integrity note

The "smallest" model should be a genuinely *trained* transformer -- one that learned the CA function from data via gradient descent. Approaches that hardcode the computation inside frozen transformer modules (using attention as a routing mechanism with hand-set weights and zero trainable parameters) satisfy the letter of the verifier checks but not the spirit of the challenge. If you find such an approach, document it in your report, but also find the smallest *trained* model that achieves the accuracy target.

Go.
