import os
import tempfile
from unittest.mock import MagicMock

import torch

from src.loggers.metric_entry import MetricEntry
from src.loggers.json_logger import JsonLogger
from src.utils.checkpoint import CheckpointManager, CheckpointState


def test_metrics_logger():
    """Tests writing and reading metrics with JsonLogger."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp:
        path = tmp.name

    logger = JsonLogger(path)
    entry1 = MetricEntry(
        iter_num=0,
        train={"loss": 1.5, "lr": 1e-4, "grad_norm": 0.5, "avg_grad_norm": 0.45},
        val={"loss": 1.6},
        system={"time_ms": 100.0, "tokens_processed": 1000},
    )
    entry2 = MetricEntry(
        iter_num=1,
        train={"loss": 1.4, "lr": 1e-4, "grad_norm": 0.4, "avg_grad_norm": 0.425},
        val={"loss": 1.5},
        system={"time_ms": 102.0, "tokens_processed": 2000},
    )

    logger.log(entry1)
    logger.log(entry2)

    read_entries = logger.read_all()
    assert len(read_entries) == 2
    assert read_entries[0] == entry1
    assert read_entries[1] == entry2

    os.remove(path)


def test_checkpoint_manager():
    """Tests saving and loading latest and best checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # CheckpointManager.cfg is only used for wandb artifacts; mock it out.
        manager = CheckpointManager(
            output_dir=tmpdir,
            cfg=MagicMock(),
            wandb_is_enabled=False,
        )

        # --- State 1: Initial save ---
        state1 = CheckpointState(
            model_state_dict={"weight": torch.tensor([1.0])},
            optimizer_state_dict={"lr": 0.1},
            iter_num=100,
            val_loss=1.5,
            config={"param": "value"},
        )
        best_loss = manager.save(state1, current_best_val_loss=float("inf"))
        assert best_loss == 1.5

        loaded_latest = manager.load("latest")
        loaded_best = manager.load("best")
        assert loaded_latest is not None and loaded_latest.iter_num == 100
        assert loaded_best is not None and loaded_best.iter_num == 100

        # --- State 2: Worse loss ---
        state2 = CheckpointState(
            model_state_dict={"weight": torch.tensor([2.0])},
            optimizer_state_dict={"lr": 0.2},
            iter_num=200,
            val_loss=1.6,
            config={"param": "value"},
        )
        best_loss = manager.save(state2, current_best_val_loss=best_loss)
        assert best_loss == 1.5

        loaded_latest = manager.load("latest")
        loaded_best = manager.load("best")
        assert loaded_latest is not None and loaded_latest.iter_num == 200
        assert loaded_best is not None and loaded_best.iter_num == 100

        # --- State 3: Better loss ---
        state3 = CheckpointState(
            model_state_dict={"weight": torch.tensor([3.0])},
            optimizer_state_dict={"lr": 0.3},
            iter_num=300,
            val_loss=1.4,
            config={"param": "value"},
        )
        best_loss = manager.save(state3, current_best_val_loss=best_loss)
        assert best_loss == 1.4

        loaded_latest = manager.load("latest")
        loaded_best = manager.load("best")
        assert loaded_latest is not None and loaded_latest.iter_num == 300
        assert loaded_best is not None and loaded_best.iter_num == 300
