"""
Configuration constants.

This module centralizes model and training hyperparameters so they can be
imported from other modules. Values are defined as module-level constants and
are intentionally lightweight (no runtime logic beyond device selection).
"""

import torch


# --- Run Details ---
RUN_NAME_PREFIX: str = "shakespeare_baseline_L4_E64"
SEED: int = 1337

# --- Model Parameters ---
BLOCK_SIZE: int = 64  # Maximum context length for predictions
N_EMBD: int = 64  # Embedding dimension
N_HEAD: int = 4  # Number of attention heads
N_LAYER: int = 4  # Number of transformer blocks
DROPOUT: float = 0.0  # Dropout rate (0 means no dropout)

# --- Training Parameters ---
BATCH_SIZE: int = 64  # How many independent sequences to process in parallel
LEARNING_RATE: float = 3e-4  # Learning rate for the optimizer
MIN_LR: float = 3e-5  # Minimum learning rate after decay
WEIGHT_DECAY: float = 1e-1  # Weight decay for regularization
MAX_ITERS: int = 10000  # Total number of training iterations
LR_DECAY_ITERS: int = (
    MAX_ITERS  # No. iters to decay learning rate (should be <= MAX_ITERS)
)
WARMUP_ITERS: int = 0  # No. iters for learning rate warmup (should be <= MAX_ITERS)
USE_COSINE_LR: bool = True  # Whether to use cosine learning rate decay
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
CHECKPOINT_PARENT_DIR: str = "checkpoints"
CHECKPOINT_FILE_PREFIX: str = "ckpt"
LOG_INTERVAL: int = 100  # How often to log training progress to the console
