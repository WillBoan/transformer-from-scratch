"""
Weights & Biases logger integration.
"""

from typing import Any, Literal, cast
import wandb
from omegaconf import DictConfig, OmegaConf

from .metric_entry import MetricEntry
from src.config import TransformerConfig


class WandbLogger:
    def __init__(
        self,
        project: str,
        group: str,
        run_name: str,
        config: DictConfig | TransformerConfig,
        mode: Literal["online", "offline", "disabled"],
    ) -> None:
        """
        Initializes the Weights & Biases logger.

        Args:
            project: Name of the W&B project.
            group: Name of the W&B group.
            run_name: Name for this specific run.
            config: Hydra configuration object to log as hyperparameters.
            mode: W&B mode, can be "online", "offline", or "disabled".
        """
        raw_config = OmegaConf.to_container(config, resolve=True)
        config_dict = cast(dict[str, Any], raw_config)

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
