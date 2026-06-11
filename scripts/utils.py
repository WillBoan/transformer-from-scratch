import os
from datetime import datetime
import config as current_cfg


def get_run_dir(run_id: str | None = None) -> str:
    """Get the run directory for the current training run."""
    if run_id is not None:
        return os.path.join(current_cfg.CHECKPOINT_PARENT_DIR, run_id)
    else:
        # Create a new run directory with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_run_dir = os.path.join(current_cfg.CHECKPOINT_PARENT_DIR, f"run_{timestamp}")
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir


def get_latest_run_dir() -> tuple[str, str]:
    """Get the latest run directory based on timestamp in the name."""
    if not os.path.isdir(current_cfg.CHECKPOINT_PARENT_DIR):
        raise FileNotFoundError(
            f"Checkpoints parent directory not found: {current_cfg.CHECKPOINT_PARENT_DIR}"
        )

    # List all subdirectories and filter those that match the expected pattern
    subdirs = [
        d
        for d in os.listdir(current_cfg.CHECKPOINT_PARENT_DIR)
        if os.path.isdir(os.path.join(current_cfg.CHECKPOINT_PARENT_DIR, d)) and "_" in d
    ]

    if not subdirs:
        raise FileNotFoundError("No valid run directories found.")

    # Sort by timestamp extracted from the directory name
    # (assuming format 'YYYY-MM-DD_HH-MM-SS_...')
    subdirs.sort(key=lambda x: x.split("_")[0] + "_" + x.split("_")[1], reverse=True)

    # Take the latest one
    latest_run_dir = os.path.join(current_cfg.CHECKPOINT_PARENT_DIR, subdirs[0])

    run_id = subdirs[0]

    return latest_run_dir, run_id
