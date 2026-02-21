from train import Trainer


def test_training_loop():
    trainer = Trainer(resume=False)
    trainer.train(max_iters=10)


def test_checkpoint_saving():
    trainer = Trainer(resume=False)
    trainer.train(max_iters=5)  # Train for a few iterations to create a checkpoint

    # Check that the checkpoint files were created
    import os
    from config import CHECKPOINT_DIR, CHECKPOINT_FILE_PREFIX

    latest_path = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_FILE_PREFIX}_latest.pt")
    best_path = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_FILE_PREFIX}_best.pt")

    assert os.path.exists(latest_path), "Latest checkpoint was not saved."
    assert os.path.exists(best_path), "Best checkpoint was not saved."


def test_checkpoint_loading():
    trainer1 = Trainer(resume=False)
    trainer1.train(max_iters=5)  # Train for a few iterations to create a checkpoint

    # Create a new trainer instance and load from checkpoint
    trainer2 = Trainer(resume=True)

    # Check that the iteration number was loaded correctly
    assert (
        trainer2.iter_num == trainer1.iter_num
    ), "Checkpoint did not load the correct iteration number."

    # Check that the best validation loss was loaded correctly
    assert (
        trainer2.best_val_loss == trainer1.best_val_loss
    ), "Checkpoint did not load the correct best validation loss."
