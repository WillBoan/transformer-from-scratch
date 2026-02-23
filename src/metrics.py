import os
import threading
import json
from dataclasses import dataclass


@dataclass
class MetricEntry:
    """
    Structured representation of a single metric entry.

    Properties:
    - iter_num: The training iteration number at which the metrics were recorded.
    - time_ms: The time taken for the evaluation step, in milliseconds.
    - tokens_processed: Cumulative number of tokens seen by the model.
    - train_loss: The training loss at this iteration.
    - val_loss: The validation loss at this iteration.
    - lr: The learning rate at this iteration.
    - avg_grad_norm: Average gradient norm over the evaluation interval.
    - update_to_weight_ratio: Ratio of update magnitude to weight magnitude,
        for the lm_head.
    """

    iter_num: int
    time_ms: float
    tokens_processed: int
    train_loss: float
    val_loss: float
    lr: float
    avg_grad_norm: float
    update_to_weight_ratio: float


class MetricsLogger:
    """
    Simple thread-safe JSONL metrics logger.
    Use: metrics.log(MetricEntry(iter=i, train_loss=..., ...))
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        # open lazily per write to be robust to crashes and multiprocess use

    def log(self, metrics: MetricEntry) -> None:
        line = json.dumps(metrics.__dict__, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[MetricEntry]:
        with open(self.path, "r", encoding="utf-8") as f:
            return [MetricEntry(**json.loads(log)) for log in f]
