from typing import Literal, Iterator
from logging import Logger
import os
import time
import math
from itertools import cycle
from contextlib import nullcontext

import torch
from torch import Tensor

from src.config import TransformerConfig
from src.models.transformer import Transformer
from src.utils.data import create_dataloader
from src.utils.tokenizer import CharTokenizer
from src.utils.checkpoint import CheckpointManager, CheckpointState
from src.utils.device_type import get_device_type
from src.utils.run_name import generate_run_name
from src.loggers.logging_config import setup_logging
from src.loggers.metric_entry import MetricEntry
from src.loggers.json_logger import JsonLogger
from src.loggers.wandb_logger import WandbLogger


class Trainer:
    """
    A class to encapsulate the training and evaluation loop for the Transformer model,
    driven by a Hydra configuration object.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        """
        Initializes the Trainer based on the provided Hydra configuration.
        """
        self.cfg = cfg
        self.cfg_dict = TransformerConfig.to_dict(self.cfg)
        self.output_dir: str
        self.logger: Logger
        self.wandb_logger: WandbLogger | None = None

        # --- Setup Output Directory and Local Logger ---
        self.output_dir = os.getcwd()
        self.logger = setup_logging(self.output_dir)
        self.logger.info(f"Hydra output directory: {self.output_dir}")

        # --- Generate the Run Name ---
        self.run_name = generate_run_name(cfg)
        self.logger.info(f"Run name: {self.run_name}")

        # --- Setup W&B Logger ---
        if self.cfg.tracking.mode != "disabled":
            self.wandb_logger = WandbLogger(
                project=self.cfg.experiment.project,
                group=self.cfg.experiment.group,
                run_name=self.run_name,
                config_dict=self.cfg_dict,
                mode=self.cfg.tracking.mode,
            )

        # --- Set random seed for reproducibility ---
        torch.manual_seed(self.cfg.system.seed)

        # --- Setup device and mixed precision context ---
        self.device_type = get_device_type(self.cfg.system.device)
        self.device = torch.device(self.device_type)
        self.ctx = self._get_autocast_context()
        self.scaler = torch.GradScaler(enabled=(self.device_type == "cuda"))

        # --- Initialize helpers ---
        metrics_log_path = os.path.join(self.output_dir, "metrics.jsonl")
        self.json_logger = JsonLogger(metrics_log_path)
        self.checkpoint_manager = CheckpointManager(
            self.output_dir,
            self.cfg,
            wandb_is_enabled=(self.wandb_logger is not None),
        )

        # --- Initialize Tokenizer & Data ---
        self.tokenizer = self._init_tokenizer()
        self._init_data()

        # --- Initialize model ---
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
            block_size=self.cfg.model.block_size,
            n_embd=self.cfg.model.n_embd,
            n_head=self.cfg.model.n_head,
            n_layer=self.cfg.model.n_layer,
            dropout=self.cfg.model.dropout,
        )
        self.model.to(self.device)
        self.logger.info(f"Model loaded on {self.device_type}")

        # --- Initialize optimizer ---
        self.optimizer = self.model.configure_optimizer(
            weight_decay=self.cfg.training.weight_decay,
            learning_rate=self.cfg.training.learning_rate,
            betas=(0.9, 0.95),
        )

        # --- Load from checkpoint if available ---
        self.iter_num = 0
        self.best_val_loss = float("inf")
        self._resume_from_checkpoint()

    def _get_autocast_context(self) -> torch.autocast | nullcontext:
        if self.device_type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _init_tokenizer(self) -> CharTokenizer:
        tokenizer = CharTokenizer()

        self.logger.info(f"Loading vocabulary from {self.cfg.data.vocab_path}")

        tokenizer.load(self.cfg.data.vocab_path)
        # We also save a copy to the run's output directory for perfect reproducibility.
        tokenizer.save(os.path.join(self.output_dir, "vocab.json"))

        return tokenizer

    def _init_data(self) -> None:
        train_data_path = self.cfg.data.train_data_path
        val_data_path = self.cfg.data.val_data_path

        if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
            raise FileNotFoundError(
                f"Training or validation data files not found at "
                f"{train_data_path} and {val_data_path}. "
                f"Please run the data preparation script first."
            )

        pin_memory = self.device_type == "cuda"

        # Create DataLoaders for train and val splits
        self.train_dataloader = create_dataloader(
            data_path=train_data_path,
            block_size=self.cfg.model.block_size,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,  # Shuffle TRUE for training data
            pin_memory=pin_memory,
        )
        self.val_dataloader = create_dataloader(
            data_path=val_data_path,
            block_size=self.cfg.model.block_size,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,  # Shuffle FALSE for validation data
            pin_memory=pin_memory,
        )

        # Create an infinite iterator for the training DataLoader,
        # so we can call next() on it indefinitely (in both training and evaluation).
        self.train_iterator: Iterator[tuple[Tensor, Tensor]] = cycle(
            self.train_dataloader
        )

        self.logger.info("Data loaders initialized.")

    def _get_batch(
        self,
        split: Literal["train", "val"],
        val_iterator: Iterator[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Fetches a batch of data for the given split.

        - For 'train', it uses the infinite train_iter.
        - For 'val', it uses the provided val_iterator, which should be a fresh iterator
            over the validation DataLoader.

        In either case, the returned tensors are moved to the appropriate device.
        """
        if split == "train":
            X, Y = next(self.train_iterator)
        else:
            if val_iterator is None:
                raise ValueError("val_iterator must be provided when split is 'val'")
            X, Y = next(val_iterator)

        return X.to(self.device), Y.to(self.device)

    def _resume_from_checkpoint(self) -> None:
        state: CheckpointState | None = self.checkpoint_manager.load("latest")
        if state:
            self.model.load_state_dict(state.model_state_dict)
            self.optimizer.load_state_dict(state.optimizer_state_dict)
            self.iter_num = state.iter_num

            best_state = self.checkpoint_manager.load("best")
            self.best_val_loss = best_state.val_loss if best_state else float("inf")

            self.logger.info(
                f"Resumed from checkpoint in run '{self.run_name}', "
                f"at iteration {self.iter_num}"
            )
        else:
            self.logger.info(f"Starting new run '{self.run_name}' from scratch.")

    def get_lr(self) -> float:
        """Get learning rate for a given iteration using cosine decay."""
        it = self.iter_num
        cfg = self.cfg.training
        if not cfg.use_cosine_lr:
            return cfg.learning_rate
        # 1) linear warmup for warmup_iters steps
        if cfg.warmup_iters > 0 and it < cfg.warmup_iters:
            return cfg.learning_rate * it / cfg.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    @torch.no_grad()
    def _estimate_loss(self) -> dict[str, float]:
        """Calculates the average loss over eval_iters, for train and val splits."""
        out: dict[str, float] = {}

        # Set the model to evaluation mode (disabling dropout, etc.)
        self.model.eval()

        # Create a fresh, finite iterator for the validation set
        # for this specific evaluation run.
        val_iterator = iter(self.val_dataloader)

        # Evaluate loss on both training and validation splits
        for split in ("train", "val"):
            losses = torch.zeros(self.cfg.training.eval_iters)
            for k in range(self.cfg.training.eval_iters):
                X, Y = self._get_batch(split, val_iterator)
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
            val_loss=latest_val_loss,
            config=self.cfg_dict,
        )
        # Save latest checkpoint, and also best if improved. Update self.best_val_loss.
        self.best_val_loss = self.checkpoint_manager.save(state, self.best_val_loss)

        # TODO: Also save a W&B Artifact for the checkpoint, if W&B is enabled

    def _log_metrics(self, entry: MetricEntry) -> None:
        """Log metrics to all configured loggers."""
        self.json_logger.log(entry)
        if self.wandb_logger:
            self.wandb_logger.log(entry)

    def _evaluate_and_checkpoint(
        self,
        iter_num: int,
        start_time_training: float,
        start_time_eval_interval: float,
        lr: float,
        grad_norm: float,
        avg_grad_norm: float,
        update_ratio: float,
    ) -> None:
        try:
            # Calculate time
            this_eval_time = time.time()
            time_ms_training = (this_eval_time - start_time_training) * 1000
            time_ms_eval_interval = (this_eval_time - start_time_eval_interval) * 1000

            # Estimate losses
            losses = self._estimate_loss()

            # Log to console
            self.logger.info(
                f"step {self.iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            # Calculate cumulative tokens processed
            tokens_processed = (
                iter_num * self.cfg.data.batch_size * self.cfg.model.block_size
            )
            # Calculate tokens per second (for this evaluation interval)
            tokens_per_sec = tokens_processed / (time_ms_eval_interval / 1000)

            # Log metrics (structured)
            entry = MetricEntry(
                iter_num=iter_num,
                train={
                    "loss": losses["train"],
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "avg_grad_norm": avg_grad_norm,
                    "update_ratio": update_ratio,
                },
                val={
                    "loss": losses["val"],
                },
                system={
                    "time_ms_training": time_ms_training,
                    "time_ms_eval_interval": time_ms_eval_interval,
                    "tokens_processed": tokens_processed,
                    "tokens_per_sec": tokens_per_sec,
                },
            )
            self._log_metrics(entry)

            # Save checkpoint(s)
            self._save_checkpoint(iter_num, losses["val"])
        except Exception as e:
            self.logger.error(
                f"Failed to evaluate or checkpoint at iter {iter_num}: {e}"
            )

    def _is_eval_step(self) -> bool:
        """Determines if the current iteration should trigger an evaluation."""
        return (self.iter_num + 1) % self.cfg.training.eval_interval == 0 or (
            self.iter_num + 1
        ) == self.cfg.training.max_iters

    def _train_step(
        self,
        calc_update_ratio: bool = True,
    ) -> tuple[float, float, float]:
        """
        Performs a single training step and returns loss, grad norm, and update ratio.
        """
        # Update learning rate according to schedule
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Unpack the batch of training data
        xb, yb = self._get_batch("train")

        # Forward pass
        with self.ctx:
            _logits, loss = self.model(xb, yb)

        # Backpropagation (with gradient scaling for mixed precision)
        self.scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]

        # Unscale gradients
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.training.grad_clip,
        )

        # If requested, capture the weights before the optimizer step
        update_ratio = 0.0
        if calc_update_ratio:
            target_param = self.model.lm_head.weight
            param_before = target_param.detach().clone()

        # Step the optimizer and update the scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Calculate the update-to-weight ratio
        if calc_update_ratio:
            update = target_param.detach() - param_before
            update_norm = update.norm()
            param_before_norm = param_before.norm()
            if param_before_norm > 0 and update_norm > 0:
                update_ratio = (update_norm / param_before_norm).item()
            else:
                update_ratio = 0.0

        # Zero gradients for the next step
        self.optimizer.zero_grad(set_to_none=True)

        # Return the loss, grad norm, and update ratio
        return loss.item(), grad_norm.item(), update_ratio

    def train(self) -> None:
        """Runs the main training loop."""
        self.logger.info("--- Starting Training ---")
        start_time_training = time.time()
        start_time_eval_interval = start_time_training

        running_grad_norm = 0.0
        steps_since_eval = 0

        while self.iter_num < self.cfg.training.max_iters:
            # Determine if this step will trigger an evaluation
            is_eval_step = self._is_eval_step()

            # Perform a training step
            _loss, grad_norm, update_ratio = self._train_step(
                calc_update_ratio=is_eval_step
            )

            running_grad_norm += grad_norm
            steps_since_eval += 1
            self.iter_num += 1

            # Evaluate loss and save checkpoint periodically
            if is_eval_step:
                lr = self.optimizer.param_groups[0]["lr"]
                avg_grad_norm = running_grad_norm / steps_since_eval
                self._evaluate_and_checkpoint(
                    iter_num=self.iter_num,
                    start_time_training=start_time_training,
                    start_time_eval_interval=start_time_eval_interval,
                    lr=lr,
                    grad_norm=grad_norm,
                    avg_grad_norm=avg_grad_norm,
                    update_ratio=update_ratio,
                )

                # Reset accumulators and eval interval timer
                running_grad_norm = 0.0
                steps_since_eval = 0
                start_time_eval_interval = time.time()

        self.logger.info("--- Training Complete ---")

        if self.wandb_logger:
            self.wandb_logger.finish()
