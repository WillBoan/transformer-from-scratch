from typing import Any, Literal, Final
import os
from dataclasses import dataclass
import torch
from config import CHECKPOINT_DIR, CHECKPOINT_FILE_PREFIX


@dataclass
class CheckpointState:
    """
    Structured representation of the checkpoint state.

    Args:
    - model_state_dict: The state dictionary of the model (weights and buffers).
    - optimizer_state_dict: The state dictionary of the optimizer (e.g., momentums).
    - iter_num: The training iteration number at which the checkpoint was saved.
    - latest_val_loss: The validation loss at the time of saving this checkpoint.
    - best_val_loss: The best validation loss achieved up to this checkpoint.
    - config: The training configuration (e.g., hyperparameters) at the time of saving.
    """

    model_state_dict: Final[dict[str, Any]]
    optimizer_state_dict: Final[dict[str, Any]]
    iter_num: Final[int]
    latest_val_loss: Final[float]
    best_val_loss: float | None
    config: Final[dict[str, Any]]


class CheckpointManager:
    """
    Atomic save & load for checkpoints. Saves both 'latest.pt' and 'best.pt'.
    """

    def __init__(
        self,
        checkpoint_dir: str = CHECKPOINT_DIR,
        prefix: str = CHECKPOINT_FILE_PREFIX,
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix

    def _get_checkpoint_path(
        self,
        checkpoint_type: Literal["latest", "best"],
    ) -> str:
        """Get the target path for a given checkpoint type."""
        return os.path.join(self.checkpoint_dir, f"{self.prefix}_{checkpoint_type}.pt")

    def _save(
        self,
        state: CheckpointState,
        checkpoint_type: Literal["latest", "best"] = "latest",
        atomic: bool = True,
    ) -> str:
        """Atomically save checkpoint and return path."""
        path = self._get_checkpoint_path(checkpoint_type)
        if atomic:
            tmp = path + ".tmp"
            torch.save(state.__dict__, tmp)
            os.replace(tmp, path)  # atomic on most OSes
        else:
            torch.save(state.__dict__, path)
        return path

    def save(self, state: CheckpointState) -> float:
        """
        Save 'latest' checkpoint, and also 'best' if val loss improved.

        Args:
            state (CheckpointState): The checkpoint state to save, which includes
                the latest validation loss and the best validation loss so far.

        Returns:
            best_val_loss (float): The best validation loss after saving the checkpoint
                (which may be updated, if the latest validation loss is better than the
                previous best).
        """
        # Check if the latest validation loss is a new best
        if state.best_val_loss is None or state.latest_val_loss < state.best_val_loss:
            # Important: update best_val_loss in state before saving either checkpoint
            state.best_val_loss = state.latest_val_loss
            self._save(state, checkpoint_type="best")
        self._save(state, checkpoint_type="latest")

        return state.best_val_loss

    def load(
        self,
        checkpoint_type: Literal["latest", "best"] = "latest",
    ) -> CheckpointState | None:
        path = self._get_checkpoint_path(checkpoint_type)
        if not os.path.exists(path):
            return None
        state_dict = torch.load(path, map_location="cpu")
        return CheckpointState(**state_dict)
