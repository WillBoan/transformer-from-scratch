from typing import Any
from logging import Logger
import os
import math
import time
from contextlib import nullcontext

import torch

import config as cfg
from model import Transformer
from utils import DataManager
from logging_config import setup_logging
from metrics import MetricsLogger, MetricEntry
from checkpoint_manager import CheckpointManager, CheckpointState


class Trainer:
    """
    A class to encapsulate the training and evaluation loop for the Transformer model.

    Attributes:
    - logger: A Logger instance for logging training progress and events.
    - ctx: A context manager for mixed precision training (torch.autocast)
        or a null context if not using mixed precision.
    - scaler: A GradScaler for scaling gradients when using mixed precision.
    - metrics_logger: An instance of MetricsLogger for logging training and validation
        metrics.
    - checkpoint_manager: An instance of CheckpointManager for saving and loading model
        checkpoints.
    - data_manager: An instance of DataManager for handling data loading,
        tokenization, and batching.
    - model: The Transformer model being trained.
    - optimizer: The optimizer used for training the model (e.g., AdamW).
    - iter_num: The current training iteration number.
    - best_val_loss: The best validation loss achieved during training,
        used for checkpointing.
    - config_dict: A dictionary of the training configuration (hyperparameters)
        to be saved with checkpoints.
    """

    logger: Logger
    ctx: torch.autocast | nullcontext[None]
    scaler: torch.GradScaler
    metrics_logger: MetricsLogger
    checkpoint_manager: CheckpointManager
    data_manager: DataManager
    model: Transformer
    optimizer: torch.optim.Optimizer
    iter_num: int
    best_val_loss: float | None
    config_dict: dict[str, Any]

    def __init__(self, resume: bool = True) -> None:
        """
        Initializes the Trainer, setting up the model, optimizer, and data.

        Args:
            resume (bool): If True, resume training from the latest checkpoint
                if available; otherwise start from scratch. Defaults to True.
        """
        self.logger = setup_logging()

        torch.manual_seed(cfg.SEED)  # pyright: ignore[reportUnknownMemberType]

        # --- Setup directories, device context, and scaler ---
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        self.ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if cfg.DEVICE == "cuda"
            else nullcontext()
        )
        if cfg.DEVICE == "mps":
            self.logger.warning(
                "Mixed precision is not supported on MPS. Using full precision."
            )
        self.scaler = torch.GradScaler(enabled=(cfg.DEVICE == "cuda"))

        # --- Initialize helpers ---
        self.metrics_logger = MetricsLogger(cfg.TRAINING_LOG_FILE)
        self.checkpoint_manager = CheckpointManager()

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
        self.logger.info(f"Model loaded on {cfg.DEVICE}")

        # --- Initialize optimizer ---
        self.optimizer = self.model.configure_optimizer(
            weight_decay=cfg.WEIGHT_DECAY,
            learning_rate=cfg.LEARNING_RATE,
            betas=(0.9, 0.95),
        )

        # --- Training state ---
        self.iter_num = 0
        self.best_val_loss: float | None = None
        self.config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}

        # --- Resume from checkpoint if available ---
        if resume:
            state: CheckpointState | None = self.checkpoint_manager.load("latest")
            if state:
                self.model.load_state_dict(state.model_state_dict)
                self.optimizer.load_state_dict(state.optimizer_state_dict)
                self.iter_num = state.iter_num
                self.best_val_loss = state.best_val_loss
                self.logger.info(
                    f"Resumed from checkpoint at iteration {self.iter_num}"
                )
            else:
                self.logger.info("No checkpoint found, starting from scratch.")

    def get_lr(self, it: int) -> float:
        """Get learning rate for a given iteration using cosine decay."""
        if not cfg.USE_COSINE_LR:
            return cfg.LEARNING_RATE
        # 1) linear warmup for warmup_iters steps
        if it < cfg.WARMUP_ITERS:
            return cfg.LEARNING_RATE * it / cfg.WARMUP_ITERS
        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.LR_DECAY_ITERS:
            return cfg.MIN_LR
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - 0) / (cfg.LR_DECAY_ITERS - 0)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.MIN_LR + coeff * (cfg.LEARNING_RATE - cfg.MIN_LR)

    @torch.no_grad()
    def _estimate_loss(self) -> dict[str, float]:
        """Calculates the average loss over EVAL_ITERS for train and val splits."""
        out: dict[str, float] = {}

        # Set the model to evaluation mode (disabling dropout, etc.)
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

        # Restore training mode
        self.model.train()

        return out

    def _save_checkpoint(self, iter_num: int, latest_val_loss: float) -> None:
        state = CheckpointState(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            iter_num=iter_num,
            latest_val_loss=latest_val_loss,
            best_val_loss=self.best_val_loss,
            config=self.config_dict,
        )
        # Save latest checkpoint, and also best if improved. Update self.best_val_loss.
        self.best_val_loss = self.checkpoint_manager.save(state)

    def _evaluate_and_checkpoint(self, start_time: float) -> None:
        try:
            losses = self._estimate_loss()
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"step {self.iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            # Log metrics (structured)
            entry = MetricEntry(
                iter_num=self.iter_num,
                train_loss=losses["train"],
                val_loss=losses["val"],
                lr=cfg.LEARNING_RATE,
                time_ms=elapsed_time * 1000.0,
            )
            self.metrics_logger.log(entry)

            # Save checkpoint(s)
            self._save_checkpoint(self.iter_num, losses["val"])
        except Exception as e:
            self.logger.error(
                f"Failed to evaluate or save checkpoint at iter {self.iter_num}: {e}"
            )

    def _train_step(self) -> tuple[float, float]:
        """Performs a single training step and returns loss and grad norm."""
        lr = self.get_lr(self.iter_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = self.data_manager.get_batch("train")
        with self.ctx:
            _logits, loss = self.model(xb, yb)

        # Backpropagation with gradient scaling for mixed precision
        self.scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]

        # Gradient clipping to prevent exploding gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), cfg.GRAD_CLIP
        )

        # Step the optimizer and update the scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Zero gradients for the next step
        self.optimizer.zero_grad(set_to_none=True)

        # Return the loss and grad norm for logging
        return loss.item(), grad_norm.item()

    def train(self, max_iters: int = cfg.MAX_ITERS) -> None:
        """Runs the main training loop."""
        start_time = time.time()
        self.logger.info("--- Starting Training ---")
        while self.iter_num < max_iters:
            # Evaluate loss and save checkpoint periodically
            if self.iter_num % cfg.EVAL_INTERVAL == 0 or self.iter_num == max_iters - 1:
                self._evaluate_and_checkpoint(start_time)

            loss, grad_norm = self._train_step()

            # Log grad norm occasionally
            if self.iter_num % cfg.LOG_INTERVAL == 0:
                self.logger.info(
                    f"Iter {self.iter_num}: loss {loss:.4f}, grad_norm {grad_norm:.4f}"
                )

            self.iter_num += 1

        self.logger.info("--- Training Complete ---")
        if self.best_val_loss:
            self.logger.info(f"Final best validation loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    trainer = Trainer(resume=True)
    trainer.train()
