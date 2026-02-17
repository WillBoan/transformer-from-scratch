from __future__ import annotations
from typing import Final
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


# --- Hyperparameters ---
# For now, we'll keep them here. Later, we'll move them to a config file.
block_size = 8  # what is the maximum context length for predictions?
n_embd = 32  # embedding dimension


# --- Model Components ---


class Head(nn.Module):
    """One head of self-attention"""

    key: nn.Linear
    query: nn.Linear
    value: nn.Linear
    tril: Tensor  # registered buffer (causal mask)

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.head_size: Final[int] = head_size

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("tril", tril)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the self-attention head.

        Args:
            x (Tensor): Input tensor of shape (batch, time-step, channels).

        Returns:
            Tensor: Output tensor of shape (batch, time-step, head_size).
        """

        _B, T, _C = x.shape

        k: Tensor = self.key(x)  # (B, T, head_size)
        q: Tensor = self.query(x)  # (B, T, head_size)

        # Compute attention scores (scaled dot-product attention)
        wei = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)  # (B, T, T)

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Perform the weighted aggregation of the values
        v: Tensor = self.value(x)  # (B, T, head_size)
        out: Tensor = wei @ v  # (B, T, head_size)
        return out
