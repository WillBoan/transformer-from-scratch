import pytest
import torch

from config import BATCH_SIZE, BLOCK_SIZE
from utils import DataManager
from tokenizer import CharTokenizer


@pytest.fixture
def dummy_dataset(tmp_path) -> tuple[str, CharTokenizer, str]:
    """
    Fixture to create a temporary dummy text file and a fitted tokenizer.
    Ensures tests are fast and independent of the actual dataset file.
    """
    # Create a dummy string long enough to sample batches from
    text = "hello world! this is a test string for the data manager. " * 50

    # Write to a temporary file provided by pytest
    file_path = tmp_path / "dummy_input.txt"
    file_path.write_text(text, encoding="utf-8")

    # Initialize and fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    return str(file_path), tokenizer, text


def test_char_tokenizer():
    """
    Tests the CharTokenizer to ensure vocabulary building, encoding,
    and decoding work correctly.
    """
    text = "abc abc"
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    # --- Vocabulary Size Check ---
    # Unique chars: ' ', 'a', 'b', 'c'
    assert tokenizer.vocab_size == 4

    # --- Encoding/Decoding Check ---
    encoded = tokenizer.encode("cab")
    assert len(encoded) == 3

    decoded = tokenizer.decode(encoded)
    assert decoded == "cab"

    # --- Mapping Consistency Check ---
    for ch in set(text):
        idx = tokenizer._stoi[ch]  # pyright: ignore[reportPrivateUsage]
        assert tokenizer._itos[idx] == ch  # pyright: ignore[reportPrivateUsage]


def test_get_batch(dummy_dataset):
    """
    Tests the get_batch method of the DataManager to ensure it
    returns batches of the correct shape, type, and content.
    """
    file_path, tokenizer, _ = dummy_dataset
    data_manager = DataManager(tokenizer=tokenizer, dataset_path=file_path)

    # Get a batch from both train and val splits
    x_train, y_train = data_manager.get_batch("train")
    x_val, y_val = data_manager.get_batch("val")

    # --- Shape Checks ---
    assert x_train.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert y_train.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert x_val.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert y_val.shape == (BATCH_SIZE, BLOCK_SIZE)

    # --- Dtype Checks ---
    assert x_train.dtype == torch.long
    assert y_train.dtype == torch.long

    # --- Content Check (y should be x shifted by one) ---
    # We can verify this by checking if the last (block_size - 1) tokens of x
    # match the first (block_size - 1) tokens of y for a given batch item.
    b = 0  # Check the first item in the batch
    assert torch.equal(x_train[b, 1:], y_train[b, :-1])


def test_data_split(dummy_dataset):
    """
    Tests the data splitting in the DataManager to ensure that the training
    and validation sets are correctly partitioned according to the specified ratio.
    """
    file_path, tokenizer, text = dummy_dataset
    data_manager = DataManager(tokenizer=tokenizer, dataset_path=file_path)

    # --- Data Length Check ---
    total_length = len(data_manager.train_data) + len(data_manager.val_data)
    assert total_length == len(text)

    # --- Split Ratio Check ---
    expected_train_length = int(0.9 * len(text))
    expected_val_length = len(text) - expected_train_length

    assert len(data_manager.train_data) == expected_train_length
    assert len(data_manager.val_data) == expected_val_length
