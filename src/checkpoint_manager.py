from typing import Any, Literal
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
    - best_val_loss: The best validation loss achieved up to this checkpoint.
    """

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    iter_num: int
    best_val_loss: float


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

    def save(
        self,
        state: CheckpointState,
        checkpoint_type: Literal["latest", "best"] = "latest",
    ) -> str:
        """Atomically save checkpoint and return path."""
        path = self._get_checkpoint_path(checkpoint_type)
        tmp = path + ".tmp"
        torch.save(state.__dict__, tmp)
        os.replace(tmp, path)  # atomic on most OSes
        return path

    def save_if_best(
        self,
        state: CheckpointState,
        current_val: float,
        best_val: float | None,
    ) -> float:
        """Save 'best' if current_val < best_val. Returns new best_val."""
        self.save(state, checkpoint_type="latest")
        if best_val is None or current_val < best_val:
            self.save(state, checkpoint_type="best")
            best_val = current_val
        return best_val

    def load(
        self,
        checkpoint_type: Literal["latest", "best"] = "latest",
    ) -> CheckpointState | None:
        path = self._get_checkpoint_path(checkpoint_type)
        if not os.path.exists(path):
            return None
        state_dict = torch.load(path, map_location="cpu")
        return CheckpointState(**state_dict)
