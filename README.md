# transformer-from-scratch

A decoder-only transformer implemented from scratch in PyTorch, wrapped in a
production-grade training harness: Hydra-configured experiments, a cosine LR
schedule with warmup, checkpointing, Weights & Biases + JSON logging,
reproducible runs, and tests + CI.

The model architecture follows the canonical "build GPT from first principles"
lineage (à la Karpathy's nanoGPT) — character-level, trained on Tiny Shakespeare.
The point of the project isn't a novel model; it's (1) understanding the
internals well enough to build them with only primitive PyTorch ops, and (2)
engineering the implementation to the standard I'd hold production code to.

## What's built from scratch vs. what's framework

**From scratch** (only `nn.Linear` / `nn.Embedding` / tensor ops — no
`nn.Transformer`, `nn.MultiheadAttention`, or `F.scaled_dot_product_attention`):

- Scaled dot-product self-attention with a causal mask (`Head`)
- Multi-head attention via parallel heads + output projection (`MultiHeadAttention`)
- Position-wise feed-forward block (`FeedForward`)
- Pre-norm transformer block with residual connections (`Block`)
- Token + learned positional embeddings, final layer norm, and the LM head (`Transformer`)
- Weight initialization, the AdamW optimizer wiring, and temperature-controlled sampling (`generate`)

**Framework, used deliberately** (and understood): autograd (`loss.backward()`),
the AdamW implementation, `F.cross_entropy`, and `DataLoader`-style batching.

## Engineering

- **Config-driven** — all hyperparameters live in `configs/` and are managed by
  [Hydra](https://hydra.cc); override anything from the CLI, no magic numbers in code.
- **Reproducible** — seeded runs; each run snapshots its exact config and vocab so
  generation reconstructs the model that produced a checkpoint.
- **Training loop** — cosine LR schedule with warmup, gradient clipping, periodic
  evaluation on a held-out split, and best/latest checkpointing.
- **Logging** — pluggable loggers (Weights & Biases for remote tracking, a JSON
  logger for local runs), toggled by config.
- **Data** — a character tokenizer and a memory-efficient pretokenized binary
  dataset (`uint16` `.bin` files).
- **Typed throughout**, with unit tests (`tests/`) and a CI workflow.

## Quickstart

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Prepare data (downloads Tiny Shakespeare, builds vocab, tokenizes to .bin)
python scripts/prepare_data.py

# 3. Train (Hydra-configured; W&B on by default)
python train.py
python train.py tracking=none                          # disable remote tracking
python train.py model.n_layer=8 training.learning_rate=1e-3   # override any config

# 4. Generate from a trained checkpoint (defaults to the latest run)
python scripts/generate.py --prompt "O Romeo, Romeo!" --tokens 200 --temperature 0.8

# 5. Plot loss curves from a run's metrics
python scripts/plot.py
```

## Project structure

```
train.py                     # Hydra entrypoint
configs/                     # Hydra configs (model, training, data, tracking)
src/
├── models/transformer.py    # the from-scratch architecture
├── trainers/base_trainer.py # training/eval loop, checkpointing, logging
├── loggers/                 # W&B + JSON loggers
└── utils/                   # tokenizer, data, checkpoints, device, run naming
scripts/                     # prepare_data, generate, plot
tests/                       # unit tests
```

## Configuration

Key knobs (see `configs/config.yaml`): `model.{block_size, n_embd, n_head,
n_layer, dropout}`, `training.{learning_rate, max_iters, warmup_iters,
use_cosine_lr, grad_clip, eval_interval}`, and `system.{seed, device}` (auto-detects
CUDA / MPS / CPU).

## Status & roadmap

The implementation and training harness are complete and runnable. Planned next
steps, in rough order:

- A controlled **experiment with a written report** (e.g. depth vs. width on this
  dataset) — the harness is built for this; the analysis is the remaining piece.
- A modern architectural component implemented from scratch (**RoPE** and/or a
  **KV-cache** for generation).
- Byte-pair tokenization as an alternative to the character-level vocab.
