import os
import tempfile
import torch

from utils.checkpoint import CheckpointManager, CheckpointState
from src.logging.metric_entry import MetricEntry
from src.logging.json_logger import MetricsLogger


def test_metrics_logger():
    """Tests writing and reading metrics with MetricsLogger."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp:
        path = tmp.name

    logger = MetricsLogger(path)
    entry1 = MetricEntry(
        iter_num=0,
        train={
            "loss": 1.5,
            "lr": 1e-4,
            "grad_norm": 0.5,
            "avg_grad_norm": 0.45,
        },
        val={
            "loss": 1.6,
        },
        system={
            "time_ms": 100.0,
            "tokens_processed": 1000,
        },
    )
    entry2 = MetricEntry(
        iter_num=1,
        train={
            "loss": 1.4,
            "lr": 1e-4,
            "grad_norm": 0.4,
            "avg_grad_norm": 0.425,
        },
        val={
            "loss": 1.5,
        },
        system={
            "time_ms": 102.0,
            "tokens_processed": 2000,
        },
    )

    # Test logging
    logger.log(entry1)
    logger.log(entry2)

    # Test reading
    read_entries = logger.read_all()
    assert len(read_entries) == 2
    assert read_entries[0] == entry1
    assert read_entries[1] == entry2

    os.remove(path)


def test_checkpoint_manager():
    """Tests saving and loading latest and best checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)

        # --- State 1: Initial save ---
        state1 = CheckpointState(
            model_state_dict={"weight": torch.tensor([1.0])},
            optimizer_state_dict={"lr": 0.1},
            iter_num=100,
            latest_val_loss=1.5,
            best_val_loss=None,
            config={"param": "value"},
        )
        best_loss = manager.save(state1)
        assert best_loss == 1.5

        # Verify 'latest' and 'best' were saved
        loaded_latest = manager.load("latest")
        loaded_best = manager.load("best")
        assert loaded_latest is not None and loaded_latest.iter_num == 100
        assert loaded_best is not None and loaded_best.iter_num == 100
        assert loaded_best.best_val_loss == 1.5

        # --- State 2: Worse loss ---
        state2 = CheckpointState(
            model_state_dict={"weight": torch.tensor([2.0])},
            optimizer_state_dict={"lr": 0.2},
            iter_num=200,
            latest_val_loss=1.6,  # Worse than 1.5
            best_val_loss=best_loss,
            config={"param": "value"},
        )
        best_loss = manager.save(state2)
        assert best_loss == 1.5  # Best loss should not change

        # Verify 'latest' is updated, but 'best' is not
        loaded_latest = manager.load("latest")
        loaded_best = manager.load("best")
        assert loaded_latest is not None and loaded_latest.iter_num == 200
        assert (
            loaded_best is not None and loaded_best.iter_num == 100
        )  # Still the first one

        # --- State 3: Better loss ---
        state3 = CheckpointState(
            model_state_dict={"weight": torch.tensor([3.0])},
            optimizer_state_dict={"lr": 0.3},
            iter_num=300,
            latest_val_loss=1.4,  # Better than 1.5
            best_val_loss=best_loss,
            config={"param": "value"},
        )
        best_loss = manager.save(state3)
        assert best_loss == 1.4  # Best loss should update

        # Verify 'latest' and 'best' are both updated
        loaded_latest = manager.load("latest")
        loaded_best = manager.load("best")
        assert loaded_latest is not None and loaded_latest.iter_num == 300
        assert loaded_best is not None and loaded_best.iter_num == 300
