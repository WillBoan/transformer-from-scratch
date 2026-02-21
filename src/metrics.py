import os
import threading
import json
from dataclasses import dataclass
import config as cfg


@dataclass
class MetricEntry:
    """
    Structured representation of a single metric entry.

    Args:
    - iter_num: The training iteration number at which the metrics were recorded.
    - train_loss: The training loss at this iteration.
    - val_loss: The validation loss at this iteration.
    - lr: The learning rate at this iteration.
    - time_ms: The time taken for the evaluation step, in milliseconds.
    """

    iter_num: int
    train_loss: float
    val_loss: float
    lr: float
    time_ms: float


class MetricsLogger:
    """
    Simple thread-safe JSONL metrics logger.
    Use: metrics.log(MetricEntry(iter=i, train_loss=..., ...))
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or cfg.TRAINING_LOG_FILE
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
