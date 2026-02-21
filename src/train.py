from contextlib import nullcontext
import os
import time

import torch

import config as cfg
from model import Transformer
from utils import DataManager
from logging_config import setup_logging
from metrics import MetricsLogger, MetricEntry
from checkpoint_manager import CheckpointManager, CheckpointState

logger = setup_logging()


class Trainer:
    """
    A class to encapsulate the training and evaluation loop for the Transformer model.

    Attributes:
    - ctx: A context manager for mixed precision training (torch.autocast)
        or a null context if not using mixed precision.
    - data_manager: An instance of DataManager for handling data loading,
        tokenization, and batching.
    - model: The Transformer model being trained.
    - optimizer: The optimizer used for training the model (e.g., AdamW).
    """

    ctx: torch.autocast | nullcontext[None]
    data_manager: DataManager
    model: Transformer
    optimizer: torch.optim.Optimizer

    def __init__(self, resume: bool = True) -> None:
        """
        Initializes the Trainer, setting up the model, optimizer, and data.

        Args:
            resume (bool): If True, resume training from the latest checkpoint
                if available; otherwise start from scratch. Defaults to True.
        """
        torch.manual_seed(cfg.SEED)  # pyright: ignore[reportUnknownMemberType]

        # Set up directories and device context
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        if cfg.DEVICE == "cuda":
            self.ctx = torch.autocast(device_type="cuda")
        else:
            self.ctx = nullcontext()

        # Initialize helpers
        self.metrics = MetricsLogger(cfg.TRAINING_LOG_FILE)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=cfg.CHECKPOINT_DIR,
            prefix=cfg.CHECKPOINT_FILE_PREFIX,
        )

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
        logger.info(f"Model loaded on {cfg.DEVICE}")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
        )

        # Training state
        self.iter_num = 0
        self.best_val_loss: float | None = None

        # Optionally resume from latest checkpoint
        if resume:
            state: CheckpointState | None = self.checkpoint_manager.load("latest")
            if state is not None:
                logger.info(f"Resuming from checkpoint iter={state.iter_num}")
                self.model.load_state_dict(state.model_state_dict)
                self.optimizer.load_state_dict(state.optimizer_state_dict)
                self.iter_num = state.iter_num + 1
                self.best_val_loss = state.best_val_loss
            else:
                logger.info("No checkpoint found, starting from scratch.")

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
        # Compute if the latest is a new best (lower is better)
        if self.best_val_loss is None or latest_val_loss < self.best_val_loss:
            self.best_val_loss = latest_val_loss

        # Prepare checkpoint state
        state = CheckpointState(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            iter_num=iter_num,
            latest_val_loss=latest_val_loss,
            best_val_loss=self.best_val_loss,
        )

        # Save (latest, and best if improved)
        self.checkpoint_manager.save(state, checkpoint_type="latest")
        if latest_val_loss == self.best_val_loss:
            self.checkpoint_manager.save(state, checkpoint_type="best")

    def _evaluate_and_checkpoint(self, start_time: float) -> None:
        losses = self._estimate_loss()
        elapsed_time = time.time() - start_time
        logger.info(
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
        self.metrics.log(entry)

        # Save checkpoint(s)
        self._save_checkpoint(self.iter_num, losses["val"])

    def _train_step(self) -> None:
        """Performs a single training step (forward, backward, optimize)."""
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

    def train(self) -> None:
        """Runs the main training loop."""
        start_time = time.time()
        logger.info("--- Starting Training ---")

        while self.iter_num < cfg.MAX_ITERS:
            # Evaluate loss and save checkpoint periodically
            if (
                self.iter_num % cfg.EVAL_INTERVAL == 0
                or self.iter_num == cfg.MAX_ITERS - 1
            ):
                self._evaluate_and_checkpoint(start_time)

            # Get a batch and perform a training step
            self._train_step()

            self.iter_num += 1

        logger.info("--- Training Complete ---")
        logger.info(f"Final best validation loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    trainer = Trainer(resume=True)
    trainer.train()
