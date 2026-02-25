from typing import Any, Literal, Final
import os
from dataclasses import dataclass
import torch


@dataclass
class CheckpointState:
    """
    Structured representation of the checkpoint state.

    Args:
    - model_state_dict: The state dictionary of the model (weights and buffers).
    - optimizer_state_dict: The state dictionary of the optimizer (e.g., momentums).
    - iter_num: The training iteration number at which the checkpoint was saved.
    - val_loss: The validation loss at the time of this checkpoint.
    - config: The training configuration (e.g., hyperparameters) at the time of saving.
    """

    model_state_dict: Final[dict[str, Any]]
    optimizer_state_dict: Final[dict[str, Any]]
    iter_num: Final[int]
    val_loss: Final[float]
    config: Final[dict[str, Any]]


class CheckpointManager:
    """
    Atomic save & load for checkpoints. Saves both 'latest.pt' and 'best.pt'.
    """

    CHECKPOINT_FILE_PREFIX = "ckpt"

    def __init__(
        self,
        output_dir: str,
    ) -> None:
        """
        Args:
            output_dir: The directory where checkpoints will be saved, typically the
                        Hydra run directory provided by os.getcwd().
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def _get_checkpoint_path(
        self,
        checkpoint_type: Literal["latest", "best"],
    ) -> str:
        """Get the target path for a given checkpoint type."""
        return os.path.join(
            self.output_dir,
            f"{self.CHECKPOINT_FILE_PREFIX}_{checkpoint_type}.pt",
        )

    def _save(
        self,
        state: CheckpointState,
        checkpoint_type: Literal["latest", "best"] = "latest",
        atomic: bool = True,
    ) -> str:
        """Atomically save checkpoint and return path."""
        path = self._get_checkpoint_path(checkpoint_type)
        if atomic:
            tmp_path = path + ".tmp"
            torch.save(state.__dict__, tmp_path)
            os.replace(tmp_path, path)  # atomic on most OSes
        else:
            torch.save(state.__dict__, path)
        return path

    def save(self, state: CheckpointState, current_best_val_loss: float) -> float:
        """
        Save 'latest' checkpoint, and also 'best' if val loss improved.

        Args:
            state: The checkpoint state to save.
            current_best_val_loss: The best loss known by the Trainer before this save.

        Returns:
            new_best_val_loss (float): The best validation loss after saving
                (which may be updated, if the latest validation loss is better than the
                previous best).
        """
        new_best_val_loss = current_best_val_loss
        is_best = state.val_loss < current_best_val_loss

        if is_best:
            new_best_val_loss = state.val_loss
            self._save(state, checkpoint_type="best")

        self._save(state, checkpoint_type="latest")

        return new_best_val_loss

    def load(
        self,
        checkpoint_type: Literal["latest", "best"] = "latest",
    ) -> CheckpointState | None:
        path = self._get_checkpoint_path(checkpoint_type)
        if not os.path.exists(path):
            return None
        state_dict = torch.load(path, map_location="cpu")
        return CheckpointState(**state_dict)
