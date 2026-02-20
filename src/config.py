"""
Configuration constants.

This module centralizes model and training hyperparameters so they can be
imported from other modules. Values are defined as module-level constants and
are intentionally lightweight (no runtime logic beyond device selection).
"""

import torch


# --- Model Parameters ---
block_size: int = 8  # Maximum context length for predictions
n_embd: int = 32  # Embedding dimension
n_head: int = 4  # Number of attention heads
n_layer: int = 2  # Number of transformer blocks
dropout: float = 0.0  # Dropout rate (0 means no dropout)

# --- Training Parameters ---
batch_size: int = 64  # How many independent sequences will we process in parallel?
learning_rate: float = 3e-4
max_iters: int = 5000
eval_interval: int = 500
eval_iters: int = 200
# Prefer CUDA, then MPS, else CPU
device: str = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# --- Checkpointing ---
out_dir: str = "checkpoints"
log_file: str = "logs.jsonl"
