from typing import Any
from dataclasses import dataclass, field, asdict


@dataclass
class MetricEntry:
    """
    Structured representation of a single metric entry, organized by namespace.
    """

    iter_num: int
    train: dict[str, Any] = field(default_factory=dict)
    val: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Flattens the metric entry into a single dictionary with namespace prefixes,
        suitable for logging to W&B or other flat key-value stores.

        Example: {"train/loss": 0.1, "system/tokens_per_sec": 5000}
        """
        flat_dict = {}
        for namespace, metrics in asdict(self).items():
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    flat_dict[f"{namespace}/{key}"] = value

        # iter_num is the step, not a metric to be namespaced
        flat_dict["iter_num"] = self.iter_num
        return flat_dict
