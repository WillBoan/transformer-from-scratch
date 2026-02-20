"""
Configuration constants.

This module centralizes model and training hyperparameters so they can be
imported from other modules. Values are defined as module-level constants and
are intentionally lightweight (no runtime logic beyond device selection).
"""

import torch


# --- Model Parameters ---
BLOCK_SIZE: int = 8  # Maximum context length for predictions
N_EMBD: int = 32  # Embedding dimension
N_HEAD: int = 4  # Number of attention heads
N_LAYER: int = 2  # Number of transformer blocks
DROPOUT: float = 0.0  # Dropout rate (0 means no dropout)

# --- Training Parameters ---
BATCH_SIZE: int = 64  # How many independent sequences to process in parallel
LEARNING_RATE: float = 3e-4  # Learning rate for the optimizer
MAX_ITERS: int = 5000  # Total number of training iterations
EVAL_INTERVAL: int = 500  # How often to evaluate the model on train and val sets
EVAL_ITERS: int = 200  # Number of iterations to evaluate for each split (train and val)
GRAD_CLIP: float = 1.0  # Clip gradients at this value
# Prefer CUDA, then MPS, else CPU
DEVICE: str = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# --- Data and Checkpointing ---
DATASET_PATH: str = "data/tinyshakespeare/input.txt"
CHECKPOINT_DIR: str = "checkpoints"
TRAINING_LOG_FILE: str = "logs.jsonl"
