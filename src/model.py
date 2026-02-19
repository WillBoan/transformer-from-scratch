from __future__ import annotations
from typing import Final
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from config import (
    block_size as DEFAULT_BLOCK_SIZE,
    n_embd as DEFAULT_N_EMBD,
    dropout as DEFAULT_DROPOUT,
)


class Head(nn.Module):
    """One head of self-attention"""

    key: nn.Linear
    query: nn.Linear
    value: nn.Linear
    tril: Tensor  # registered buffer (causal mask)

    def __init__(self, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.head_size: Final[int] = head_size
        self.n_embd: Final[int] = n_embd
        self.registered_block_size: Final[int] = block_size

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

    def __init__(
        self,
        num_heads: int,
        n_embd: int,
        block_size: int,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        # Compute the head size
        assert (
            n_embd % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.n_embd: Final[int] = n_embd
        self.head_size: Final[int] = n_embd // num_heads
        self.num_heads: Final[int] = num_heads

        # Instantiate the specified number of heads and store them in a ModuleList
        self.heads = nn.ModuleList(
            [Head(self.head_size, n_embd, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Multi-Head Attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch, time-step, channels).

        Returns:
            Tensor: Output tensor of shape (batch, time-step, channels).
        """
        # Concatenate the outputs of each head along the channel dimension
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # (B, T, head_size * num_heads) = (B, T, n_embd)

        # Project the concatenated output back to the embedding dimension
        out = self.proj(out)

        # Apply dropout
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    net: nn.Sequential

    def __init__(
        self,
        n_embd: int,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
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


class Block(nn.Module):
    """
    Transformer block: communication followed by computation.

    Each block consists of a multi-head self-attention layer followed by a
    feed-forward layer, with layer normalization and residual connections
    around each of these two sub-layers.

    `nn.LayerNorm` normalizes the inputs across the features dimension. It
    helps stabilize and accelerate training by keeping the input distribution
    to each layer consistent. In Transformers, LayerNorm is applied to the
    outputs of the attention and feed-forward sub-layers to improve stability
    and convergence. Residual connections around each sub-layer let the model
    learn identity mappings, which helps mitigate vanishing gradients and
    enables deeper networks.

    Mathematically, `LayerNorm` computes the mean and variance of the input
    across the features dimension, normalizes the input, and then applies
    learnable scaling and shifting parameters. This allows the model to
    maintain the representational power while ensuring that the inputs to
    each layer have a stable distribution.

    Is this similar to how we use `softmax` to normalize attention scores
    across the sequence dimension? In both cases we normalize values to
    improve stability and convergence, but they operate on different
    dimensions and serve different purposes. `softmax` normalizes across the
    sequence dimension to produce attention weights, while `LayerNorm`
    normalizes across the features dimension to stabilize the input
    distribution for each layer.

    The "features dimension" refers to the last dimension of the input
    tensor, which corresponds to the embedding dimension in the context of
    Transformers. For example, if the input tensor has shape
    (batch_size, sequence_length, embedding_dim), LayerNorm will normalize
    across the embedding_dim dimension for each token in the sequence.

    "Residual connections" means that we add the input of a layer to its
    output before passing it to the next layer.
    """

    sa: MultiHeadAttention
    ffwd: FeedForward
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(
        self,
        num_heads: int,
        n_embd: int,
        block_size: int,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Transformer Block.

        Args:
            x (Tensor): Input tensor of shape (batch, time-step, channels).

        Returns:
            Tensor: Output tensor of shape (batch, time-step, channels).
        """
        # Self-attention with residual connection
        x = x + self.sa(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    """
    The full Transformer model.

    The Transformer model consists of an embedding layer for tokens and positions,
    followed by a stack of Transformer blocks, and finally a linear layer to
    project the output to the vocabulary size for language modeling.



    """

    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    blocks: nn.Sequential
    ln_f: nn.LayerNorm
    lm_head: nn.Linear

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        num_heads: int,
        n_embd: int | None = None,
        block_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        # use provided values or fall back to config defaults
        self.n_embd: Final[int] = n_embd if n_embd is not None else DEFAULT_N_EMBD
        self.block_size: Final[int] = (
            block_size if block_size is not None else DEFAULT_BLOCK_SIZE
        )
        self.dropout: Final[float] = dropout if dropout is not None else DEFAULT_DROPOUT
        self.num_heads: Final[int] = num_heads
        self.n_layer: Final[int] = n_layer

        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    num_heads=num_heads,
                    n_embd=self.n_embd,
                    block_size=self.block_size,
                    dropout=self.dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.n_embd)  # final layer norm
        self.lm_head = nn.Linear(self.n_embd, vocab_size)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass for the Transformer model.

        Args:
            idx (Tensor): Input tensor of token indices, shape (B, T).
            targets (Tensor | None): Target tensor of token indices, shape (B, T).

        Returns:
            tuple[Tensor, Tensor | None]: A tuple containing:
                - logits (Tensor): The model's output logits, shape (B, T, vocab_size).
                - loss (Tensor | None): The cross-entropy loss, or None if targets are
                    not provided.
        """
        B, T = idx.shape

        # Get token and position embeddings
        tok_emb: Tensor = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb: Tensor = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        # Pass through transformer blocks
        x: Tensor = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)

        # Get logits
        logits: Tensor = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape  # pyright: ignore[reportConstantRedefinition]
            logits_for_loss = logits.view(B * T, C)
            targets_for_loss = targets.view(B * T)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        """
        Generate new tokens based on a given context.

        Args:
            idx (Tensor): The initial context, an input tensor of token indices.
                - Shape: (B, T)
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            Tensor: The generated sequence of token indices.
                - Shape: (B, T + max_new_tokens)
        """
        with torch.no_grad():
            for _ in range(max_new_tokens):
                print(f"Current input shape: {idx.shape}")  # Debugging statement

                # Crop idx to the last block_size tokens
                block_size = self.block_size
                idx_cond = idx[:, -block_size:]

                # Get the model's predictions for the current input
                logits, _loss = self(idx_cond)  # logits shape: (B, T, vocab_size)

                # Focus only on the last time step
                logits = logits[:, -1, :]  # (B, vocab_size)

                # Apply softmax to get probabilities and sample from the distribution
                probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

                # Sample the next token from the probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

                # Append the predicted token to the input sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    # @overload
    # def __call__(
    #     self,
    #     idx: Tensor,
    #     targets: Tensor,
    # ) -> tuple[Tensor, Tensor]: ...

    # @overload
    # def __call__(
    #     self,
    #     idx: Tensor,
    #     targets: None = None,
    # ) -> tuple[Tensor, None]: ...

    # def __call__(
    #     self,
    #     idx: Tensor,
    #     targets: Tensor | None = None,
    # ) -> tuple[Tensor, Tensor | None]:
    #     return super().__call__(idx, targets)
