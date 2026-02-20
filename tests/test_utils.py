import torch

from config import batch_size, block_size
from utils import DataManager


def test_get_batch():
    """
    Tests the get_batch method of the DataManager to ensure it
    returns batches of the correct shape, type, and content.
    """
    # Instantiate the data manager
    data_manager = DataManager()

    # Get a batch from both train and val splits
    x_train, y_train = data_manager.get_batch("train")
    x_val, y_val = data_manager.get_batch("val")

    # --- Shape Checks ---
    assert x_train.shape == (batch_size, block_size)
    assert y_train.shape == (batch_size, block_size)
    assert x_val.shape == (batch_size, block_size)
    assert y_val.shape == (batch_size, block_size)

    # --- Dtype Checks ---
    assert x_train.dtype == torch.long
    assert y_train.dtype == torch.long

    # --- Content Check (y should be x shifted by one) ---
    # We can verify this by checking if the first (block_size - 1) tokens of y
    # match the last (block_size - 1) tokens of x for a given batch item.
    b = 0  # Check the first item in the batch
    assert torch.equal(x_train[b, 1:], y_train[b, :-1])


def test_tokenizer():
    """
    Tests the tokenizer initialization in the DataManager to ensure
    that the vocabulary size and mappings are consistent with the input text.
    """
    # Instantiate the data manager
    data_manager = DataManager()

    # --- Vocabulary Size Check ---
    unique_chars = set(data_manager.text)
    assert data_manager.vocab_size == len(unique_chars)

    # --- Mapping Consistency Check ---
    for ch in unique_chars:
        idx = data_manager._stoi[ch]  # pyright: ignore[reportPrivateUsage]
        assert data_manager._itos[idx] == ch  # pyright: ignore[reportPrivateUsage]


def test_data_split():
    """
    Tests the data splitting in the DataManager to ensure that the training
    and validation sets are correctly partitioned according to the specified ratio.
    """
    # Instantiate the data manager
    data_manager = DataManager()

    # --- Data Length Check ---
    total_length = len(data_manager.train_data) + len(data_manager.val_data)
    assert total_length == len(data_manager.text)

    # --- Split Ratio Check ---
    expected_train_length = int(0.9 * len(data_manager.text))
    expected_val_length = len(data_manager.text) - expected_train_length
    assert len(data_manager.train_data) == expected_train_length
    assert len(data_manager.val_data) == expected_val_length
