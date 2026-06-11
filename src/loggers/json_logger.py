import os
import json
import threading
from dataclasses import asdict
from .metric_entry import MetricEntry


class JsonLogger:
    """
    Simple thread-safe JSONL metrics logger.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        # open lazily per write to be robust to crashes and multiprocess use

    def log(self, metrics: MetricEntry) -> None:
        line = json.dumps(asdict(metrics), default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[MetricEntry]:
        with open(self.path, "r", encoding="utf-8") as f:
            return [MetricEntry(**json.loads(log)) for log in f]
