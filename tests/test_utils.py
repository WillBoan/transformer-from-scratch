import numpy as np
import pytest
import torch

from src.utils.data import PretokenizedDataset, create_dataloader
from src.utils.tokenizer import CharTokenizer

BLOCK_SIZE = 16
BATCH_SIZE = 4


@pytest.fixture
def dummy_bin_file(tmp_path):
    n_tokens = 300
    data = np.arange(n_tokens, dtype=np.uint16)
    bin_path = tmp_path / "data.bin"
    data.tofile(str(bin_path))
    return str(bin_path)


def test_char_tokenizer():
    text = "abc abc"
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    # Unique chars: ' ', 'a', 'b', 'c'
    assert tokenizer.vocab_size == 4

    encoded = tokenizer.encode("cab")
    assert len(encoded) == 3

    decoded = tokenizer.decode(encoded)
    assert decoded == "cab"

    for ch in set(text):
        idx = tokenizer._stoi[ch]  # pyright: ignore[reportPrivateUsage]
        assert tokenizer._itos[idx] == ch  # pyright: ignore[reportPrivateUsage]


def test_pretokenized_dataset_shape(dummy_bin_file):
    dataset = PretokenizedDataset(data_path=dummy_bin_file, block_size=BLOCK_SIZE)

    x, y = dataset[0]

    assert x.shape == (BLOCK_SIZE,)
    assert y.shape == (BLOCK_SIZE,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_pretokenized_dataset_shift(dummy_bin_file):
    """y should be x shifted forward by one token."""
    dataset = PretokenizedDataset(data_path=dummy_bin_file, block_size=BLOCK_SIZE)
    x, y = dataset[0]
    assert torch.equal(x[1:], y[:-1])


def test_pretokenized_dataset_length(dummy_bin_file):
    dataset = PretokenizedDataset(data_path=dummy_bin_file, block_size=BLOCK_SIZE)
    # len == n_tokens - block_size
    raw = np.memmap(dummy_bin_file, dtype=np.uint16, mode="r")
    assert len(dataset) == len(raw) - BLOCK_SIZE


def test_create_dataloader_shape(dummy_bin_file):
    loader = create_dataloader(
        dummy_bin_file, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    x, y = next(iter(loader))

    assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert y.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert x.dtype == torch.long
