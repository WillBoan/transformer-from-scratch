from typing import Any
from logging import Logger
import os
import json
import time
import math
from contextlib import nullcontext

import torch

import config as cfg
from model import Transformer
from utils import DataManager
from tokenizer import CharTokenizer
from logging_config import setup_logging
from metrics import MetricsLogger, MetricEntry
from checkpoint_manager import CheckpointManager, CheckpointState

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.plot import LossCurvePlotter  # noqa E402


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
    - tokenizer: An instance of CharTokenizer for encoding and decoding text.
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

    run_id: str
    checkpoint_dir: str
    logger: Logger
    ctx: torch.autocast | nullcontext[None]
    scaler: torch.GradScaler
    metrics_logger: MetricsLogger
    checkpoint_manager: CheckpointManager
    tokenizer: CharTokenizer
    data_manager: DataManager
    model: Transformer
    optimizer: torch.optim.Optimizer
    iter_num: int
    best_val_loss: float | None
    config_dict: dict[str, Any]

    def __init__(self, resume_run_id: str | None = None) -> None:
        """
        Initializes the Trainer, setting up the model, optimizer, and data.

        Args:
            resume_run_id (str | None): If provided, resume training from this
                specific run ID. Otherwise, start a new run.
        """
        # --- Setup Run ID and Directories ---
        if resume_run_id:
            self.run_id = resume_run_id
            self.checkpoint_dir = os.path.join(cfg.CHECKPOINT_PARENT_DIR, self.run_id)
        else:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.run_id = f"{time_str}_{cfg.RUN_NAME_PREFIX}"
            self.checkpoint_dir = os.path.join(cfg.CHECKPOINT_PARENT_DIR, self.run_id)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Setup Logger ---
        self.logger = setup_logging(self.checkpoint_dir)

        # --- Set random seed for reproducibility ---
        torch.manual_seed(cfg.SEED)  # pyright: ignore[reportUnknownMemberType]

        # --- Setup device context and scaler ---
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
        metrics_log_path = os.path.join(self.checkpoint_dir, "metrics.jsonl")
        self.metrics_logger = MetricsLogger(metrics_log_path)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

        # --- Initialize Tokenizer ---
        self.tokenizer = CharTokenizer()
        vocab_path = os.path.join(self.checkpoint_dir, "vocab.json")

        if resume_run_id:
            self.logger.info(f"Loading existing vocabulary from {vocab_path}")
            self.tokenizer.load(vocab_path)
        else:
            self.logger.info("Building new vocabulary from dataset...")
            with open(cfg.DATASET_PATH, "r", encoding="utf-8") as f:
                text = f.read()
            self.tokenizer.fit(text)
            self.tokenizer.save(vocab_path)

        # --- Initialize data manager ---
        self.data_manager = DataManager(
            tokenizer=self.tokenizer,
            dataset_path=cfg.DATASET_PATH,
        )

        # --- Initialize model ---
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
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
        self.config_dict = {k: v for k, v in vars(cfg).items() if k.isupper()}

        # --- Save config snapshot for new runs ---
        if not resume_run_id:
            config_path = os.path.join(self.checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(self.config_dict, f, indent=2)

        # --- Resume from checkpoint if available ---
        state: CheckpointState | None = self.checkpoint_manager.load("latest")
        if state:
            self.model.load_state_dict(state.model_state_dict)
            self.optimizer.load_state_dict(state.optimizer_state_dict)
            self.iter_num = state.iter_num
            self.best_val_loss = state.best_val_loss
            self.logger.info(
                f"Resumed from checkpoint in run '{self.run_id}', "
                f"at iteration {self.iter_num}"
            )
        else:
            self.logger.info(f"Starting new run '{self.run_id}' from scratch.")

    def get_lr(self, it: int) -> float:
        """Get learning rate for a given iteration using cosine decay."""
        if not cfg.USE_COSINE_LR:
            return cfg.LEARNING_RATE
        # 1) linear warmup for warmup_iters steps
        if cfg.WARMUP_ITERS > 0 and it < cfg.WARMUP_ITERS:
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
            # Perform a training step
            loss, grad_norm = self._train_step()

            self.iter_num += 1

            # Log grad norm periodically
            if self.iter_num % cfg.LOG_INTERVAL == 0:
                self.logger.info(
                    f"Iter {self.iter_num}: loss {loss:.4f}, grad_norm {grad_norm:.4f}"
                )

            # Evaluate loss and save checkpoint periodically
            if self.iter_num % cfg.EVAL_INTERVAL == 0 or self.iter_num == max_iters:
                self._evaluate_and_checkpoint(start_time)

        self.logger.info("--- Training Complete ---")
        if self.best_val_loss:
            self.logger.info(f"Final best validation loss: {self.best_val_loss:.4f}")

        # Plot the loss curves for the training run
        plotter = LossCurvePlotter(run_id=self.run_id)
        plotter.plot()


if __name__ == "__main__":
    trainer = Trainer(
        resume_run_id=None
    )  # Set to a specific run ID to resume, or None to start fresh
    trainer.train()
