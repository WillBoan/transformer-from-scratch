import logging
import os
import pytest

# import shutil

from train import Trainer
import config as cfg


@pytest.fixture(autouse=True)
def cleanup_logging():
    """
    Fixture to automatically clean up logging handlers after each test.
    This prevents handlers from one test from interfering with another.
    """
    yield  # Run the test
    # Teardown: remove all handlers from our specific logger
    logger = logging.getLogger("transformer_trainer")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def test_training_loop_creates_artifacts():
    """
    Test that a short training run completes and creates the expected run directory
    and artifacts.
    """
    trainer = Trainer(resume_run_id=None)
    # Use a very small number of iterations for testing
    trainer.train(max_iters=2)

    run_dir = trainer.checkpoint_dir
    assert os.path.isdir(run_dir), "Run directory was not created."

    # Check for expected files
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    log_path = os.path.join(run_dir, "logs.log")

    assert os.path.exists(config_path), "config.json was not created."
    assert os.path.exists(metrics_path), "metrics.jsonl was not created."
    assert os.path.exists(log_path), "logs.log was not created."

    # Cleanup the created directory
    # shutil.rmtree(run_dir)


def test_checkpoint_saving_and_loading():
    """
    Test that a checkpoint is saved and can be correctly loaded to resume training.
    """
    # --- First run to create a checkpoint ---
    trainer1 = Trainer(resume_run_id=None)
    trainer1.train(max_iters=5)  # Train for a few iterations

    run_id = trainer1.run_id
    run_dir = trainer1.checkpoint_dir

    # Check that the checkpoint file was created
    latest_path = os.path.join(run_dir, f"{cfg.CHECKPOINT_FILE_PREFIX}_latest.pt")
    assert os.path.exists(latest_path), "Latest checkpoint was not saved."

    # --- Second run to resume from the checkpoint ---
    trainer2 = Trainer(resume_run_id=run_id)

    # Check that the state was loaded correctly
    assert (
        trainer2.iter_num == 5
    ), "Checkpoint did not load the correct iteration number."
    assert (
        trainer2.best_val_loss == trainer1.best_val_loss
    ), "Checkpoint did not load the correct best validation loss."
    assert trainer2.run_id == run_id, "Resumed trainer has an incorrect run_id."

    # Train for a few more steps to ensure it continues without error
    trainer2.train(max_iters=10)
    assert (
        trainer2.iter_num == 10
    ), "Resumed training did not continue to the new max_iters."

    # Cleanup the created directory
    # shutil.rmtree(run_dir)
