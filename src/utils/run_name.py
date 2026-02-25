"""
Utility for generating a unique run name based on the current timestamp.

Used for logging and checkpointing to ensure that each run has a distinct identifier.
"""

import time
from src.config import TransformerConfig


def generate_run_name(cfg: TransformerConfig) -> str:
    """Generate a unique run name based on the current timestamp and configuration."""
    # If a run name is specified in the config, include it in the run name
    if cfg.experiment.run_name:
        return cfg.experiment.run_name

    # Otherwise, generate a run name based on the group, model config, and timestamp
    group = getattr(cfg.experiment, "group", "default")
    time_str = time.strftime("%Y%m%d-%H%M%S")
    run_name_parts = [
        group,
        f"L{cfg.model.n_layer}",
        f"H{cfg.model.n_head}",
        f"E{cfg.model.n_embd}",
        time_str,
    ]

    return "_".join(run_name_parts)
