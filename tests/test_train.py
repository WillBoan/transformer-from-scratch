import pytest


@pytest.mark.skip(
    reason=(
        "Trainer is Hydra-driven and requires a full TransformerConfig + real "
        "data files on disk. Integration-tested manually via samples/runs/."
    )
)
def test_training_loop_creates_artifacts():
    """
    Test that a short training run completes and creates the expected run directory
    and artifacts.
    """
    pass


@pytest.mark.skip(
    reason=(
        "Trainer is Hydra-driven and requires a full TransformerConfig + real "
        "data files on disk. Integration-tested manually via samples/runs/."
    )
)
def test_checkpoint_saving_and_loading():
    """
    Test that a checkpoint is saved and can be correctly loaded to resume training.
    """
    pass
