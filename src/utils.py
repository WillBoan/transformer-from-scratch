from typing import Literal
import torch
from torch import Tensor
from config import batch_size, block_size, device


class DataManager:
    """
    DataManager for loading, tokenizing, and batching the dataset.
    """

    file_path: str
    text: str
    vocab_size: int
    _stoi: dict[str, int]  # Mapping from characters to integers, for encoding
    _itos: dict[int, str]  # Mapping from integers to characters, for decoding
    train_data: Tensor
    val_data: Tensor

    def __init__(self, file_path: str = "data/tinyshakespeare/input.txt") -> None:
        """
        Initializes the data manager by loading and processing the dataset.

        Args:
            file_path (str): The path to the dataset file.
        """
        self.file_path = file_path

        # --- Data Loading ---
        self.load_data(self.file_path)

        # --- Tokenizer ---
        self.run_tokenizer(self.text)

        # --- Data Splitting ---
        self.split_data(self.text)

    def load_data(self, file_path: str) -> None:
        """Loads the dataset from the specified file path."""
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

    def run_tokenizer(self, text: str) -> None:
        """
        Initializes the tokenizer.

        Creates:
        - A sorted list of unique characters in the text.
        - `vocab_size`: The number of unique characters.
        - `_stoi`: A mapping from characters to integers (stoi), for encoding.
        - `_itos`: A mapping from integers to characters (itos), for decoding.
        """
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def split_data(self, text: str, split_ratio: float = 0.9) -> None:
        """Split the data into training and validation sets."""
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str) -> list[int]:
        """Encode a string into a list of integers."""
        return [self._stoi[c] for c in s]

    def decode(self, int_list: list[int]) -> str:
        """Decode a list of integers into a string."""
        return "".join([self._itos[i] for i in int_list])

    def get_batch(
        self,
        split: Literal["train", "val"],
    ) -> tuple[Tensor, Tensor]:
        """
        Generate a small batch of data of inputs x and targets y.

        Args:
            split (Literal["train", "val"]): Which data split to use.

        Returns:
            A tuple containing the input and target tensors.
        """
        # Select the appropriate data split
        data = self.train_data if split == "train" else self.val_data

        # Generate random starting indices for the batch
        ix = torch.randint(len(data) - block_size, (batch_size,))

        # Create input sequences (x) and target sequences (y)
        x = torch.stack([data[i : i + block_size] for i in ix])  # noqa E203
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # noqa E203

        # Move tensors to the configured device
        x, y = x.to(device), y.to(device)

        return x, y
