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


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.

    The outputs of each head are concatenated and projected to produce the final output
    of the multi-head attention layer.

    The multi-head attention mechanism allows the model to attend to different parts of
    the input sequence simultaneously, capturing various relationships and dependencies
    between tokens.
    """

    heads: nn.ModuleList
    proj: nn.Linear

    def __init__(self, num_heads: int) -> None:
        super().__init__()

        # Compute the head size
        self.head_size: Final[int] = n_embd // num_heads

        # Instantiate the specified number of heads and store them in a ModuleList
        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Multi-Head Attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch, time-step, channels).

        Returns:
            Tensor: Output tensor of shape (batch, time-step, channels).
        """
        # Concatenate the outputs of each head along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Project the concatenated output back to the embedding dimension
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    net: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Feed-Forward layer.

        Args:
            x (Tensor): Input tensor of shape (batch, time-step, channels).

        Returns:
            Tensor: Output tensor of shape (batch, time-step, channels).
        """
        return self.net(x)
