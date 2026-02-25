"""
Weights & Biases logger integration.
"""

from typing import Any, Literal
import wandb
from .metric_entry import MetricEntry


class WandbLogger:
    def __init__(
        self,
        project: str,
        group: str,
        run_name: str,
        config_dict: dict[str, Any],
        mode: Literal["online", "offline", "disabled"],
    ) -> None:
        """
        Initializes the Weights & Biases logger.

        Args:
            project: Name of the W&B project.
            group: Name of the W&B group.
            run_name: Name for this specific run.
            config_dict: Configuration dictionary to log as hyperparameters.
            mode: W&B mode, can be "online", "offline", or "disabled".
        """
        self.run = wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=config_dict,
            mode=mode,
        )

    def log(self, metrics: MetricEntry) -> None:
        """
        Logs a MetricEntry to W&B. The entry is flattened.
        """
        if self.run:
            flat_metrics = metrics.to_flat_dict()
            # Use iter_num as the step for W&B
            step = flat_metrics.pop("iter_num", None)
            wandb.log(flat_metrics, step=step)

    def finish(self) -> None:
        """Finishes the W&B run."""
        if self.run:
            wandb.finish()
