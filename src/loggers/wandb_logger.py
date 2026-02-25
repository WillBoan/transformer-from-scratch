"""
Weights & Biases logger integration.
"""

from typing import Literal
import wandb
from omegaconf import DictConfig, OmegaConf

from .metric_entry import MetricEntry


class WandbLogger:
    def __init__(
        self,
        project_name: str,
        run_name: str | None = None,
        config: DictConfig | None = None,
        mode: Literal["online", "offline", "disabled"] = "online",
    ) -> None:
        """
        Initializes the Weights & Biases logger.

        Args:
            project_name: Name of the W&B project.
            run_name: Name for this specific run.
            config: Hydra configuration object to log as hyperparameters.
            mode: W&B mode ("online", "offline", or "disabled").
        """
        config_dict = OmegaConf.to_container(config, resolve=True) if config else {}

        self.run = wandb.init(
            project=project_name,
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
