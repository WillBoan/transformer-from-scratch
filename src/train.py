from typing import Any
import json
import os
import time
from contextlib import nullcontext

import torch

import config as cfg
from model import Transformer
from utils import DataManager


class Trainer:
    """
    A class to encapsulate the training and evaluation loop for the Transformer model.
    """

    ctx: torch.autocast  # Context manager for mixed precision training
    data_manager: DataManager
    model: Transformer
    optimizer: torch.optim.Optimizer

    def __init__(self) -> None:
        """Initializes the Trainer, setting up the model, optimizer, and data."""
        torch.manual_seed(1337)  # pyright: ignore[reportUnknownMemberType]

        # Set up directories and device context
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        pt_dtype = torch.float16 if "cuda" in cfg.DEVICE else torch.float32
        self.ctx = torch.autocast(device_type=cfg.DEVICE, dtype=pt_dtype)

        # Initialize data manager
        self.data_manager = DataManager()

        # Initialize model
        self.model = Transformer(
            vocab_size=self.data_manager.vocab_size,
            n_layer=cfg.N_LAYER,
            n_head=cfg.N_HEAD,
            block_size=cfg.BLOCK_SIZE,
            n_embd=cfg.N_EMBD,
            dropout=cfg.DROPOUT,
        )
        self.model.to(cfg.DEVICE)
        print(f"Model loaded on {cfg.DEVICE}")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
        )

    @torch.no_grad()
    def _estimate_loss(self) -> dict[str, float]:
        """Calculates the average loss over eval_iters for train and val splits."""
        out: dict[str, float] = {}

        # Set the model to evaluation mode (disabling gradient calculations)
        self.model.eval()

        # Evaluate loss on both training and validation splits
        for split in ("train", "val"):
            losses = torch.zeros(cfg.EVAL_ITERS)
            for k in range(cfg.EVAL_ITERS):
                X, Y = self.data_manager.get_batch(split)
                with self.ctx:
                    _logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

        # Set the model back to training mode
        self.model.train()

        return out

    def train(self) -> None:
        """Runs the main training loop."""
        iter_num = 0
        best_val_loss = float("inf")
        start_time = time.time()

        print("--- Starting Training ---")
        while iter_num < cfg.MAX_ITERS:
            # Evaluate loss and save checkpoint periodically
            if iter_num % cfg.EVAL_INTERVAL == 0 or iter_num == cfg.MAX_ITERS - 1:
                losses = self._estimate_loss()
                elapsed_time = time.time() - start_time
                print(
                    f"step {iter_num}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

                # Log metrics
                log_entry: dict[str, int | float] = {
                    "iter": iter_num,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "lr": cfg.LEARNING_RATE,
                    "time_ms": elapsed_time * 1000,
                }
                with open(cfg.TRAINING_LOG_FILE, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                # Save best checkpoint
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    checkpoint: dict[str, Any] = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                    }
                    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_ckpt.pt")
                    print(f"Saving new best checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)

            # Get a batch and perform a training step
            xb, yb = self.data_manager.get_batch("train")
            with self.ctx:
                _logits, loss = self.model(xb, yb)

            # Backpropagation (compute gradients)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRAD_CLIP)

            # Optimizer step (to update model parameters)
            self.optimizer.step()

            iter_num += 1

        print("\n--- Training Complete ---")
        print(f"Final best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
