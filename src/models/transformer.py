from __future__ import annotations
from typing import Final, overload
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of self-attention"""

    head_size: int
    key: nn.Linear
    query: nn.Linear
    value: nn.Linear
    tril: Tensor  # registered buffer (causal mask)

    def __init__(
        self,
        head_size: int,
        block_size: int,
        n_embd: int,
    ) -> None:
        super().__init__()
        self.head_size: Final[int] = head_size

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        tril = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
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

    head_size: int
    heads: nn.ModuleList
    proj: nn.Linear

    def __init__(
        self,
        n_head: int,
        block_size: int,
        n_embd: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Compute the head size
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size: Final[int] = n_embd // n_head

        # Instantiate the specified number of heads and store them in a ModuleList
        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=self.head_size,
                    block_size=block_size,
                    n_embd=n_embd,
                )
                for _ in range(n_head)
            ]
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
        )  # (B, T, head_size * n_head) = (B, T, n_embd)

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
        dropout: float,
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
    A pre-norm transformer block: multi-head self-attention followed by a
    position-wise feed-forward network, each wrapped in a residual connection
    with layer normalization applied to its input.
    """

    mha: MultiHeadAttention
    ffwd: FeedForward
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(
        self,
        n_head: int,
        block_size: int,
        n_embd: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(
            n_head=n_head,
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
        x = x + self.mha(self.ln1(x))
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

    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    blocks: nn.Sequential
    ln_f: nn.LayerNorm
    lm_head: nn.Linear

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Set model parameters
        self.vocab_size: Final[int] = vocab_size
        self.block_size: Final[int] = block_size
        self.n_embd: Final[int] = n_embd
        self.n_head: Final[int] = n_head
        self.n_layer: Final[int] = n_layer
        self.dropout: Final[float] = dropout

        # Layers
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_head=self.n_head,
                    n_embd=self.n_embd,
                    block_size=self.block_size,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.n_embd)  # final layer norm
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize module weights.
        - Linear and Embedding weights are initialized from N(0, 0.02).
        - Linear biases are zeroed.
        - LayerNorm weights are ones and biases are zeros.

        Args:
            module (nn.Module): The module to initialize.
        """
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        # TODO: Add a special flag to the projection layers
        # (e.g., self.proj.NANOGPT_SCALE_INIT = 1) and update _init_weights to look for
        # it and scale the standard deviation accordingly.

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
    ) -> torch.optim.AdamW:
        """
        Configures the optimizer for the Transformer model.

        Args:
            weight_decay (float): Weight decay for regularization.
            learning_rate (float): Learning rate for the optimizer.
            betas (tuple[float, float]): Betas for the AdamW optimizer.

        Returns:
            torch.optim.AdamW: The configured optimizer.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

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

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Generate new tokens based on a given context.

        Args:
            idx (Tensor): The initial context, an input tensor of token indices.
                - Shape: (B, T)
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The sampling temperature. Higher values lead to more
                random samples, while lower values make the model more deterministic.

        Returns:
            Tensor: The generated sequence of token indices.
                - Shape: (B, T + max_new_tokens)
        """
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop idx to the last block_size tokens
                block_size = self.block_size
                idx_cond = idx[:, -block_size:]

                # Get the model's predictions for the current input
                logits, _loss = self(idx_cond)  # logits shape: (B, T, vocab_size)

                # Focus only on the last time step
                logits = logits[:, -1, :]  # (B, vocab_size)

                # Apply temperature scaling to the logits
                logits = logits / temperature

                # Apply softmax to get probabilities and sample from the distribution
                probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

                # Sample the next token from the probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

                # Append the predicted token to the input sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    @overload
    def __call__(
        self,
        idx: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def __call__(
        self,
        idx: Tensor,
        targets: None = None,
    ) -> tuple[Tensor, None]: ...

    def __call__(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        return super().__call__(idx, targets)
