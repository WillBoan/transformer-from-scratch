from typing import Literal
import torch
from torch import Tensor
from config import BATCH_SIZE, BLOCK_SIZE, DEVICE, DATASET_PATH
from tokenizer import CharTokenizer


class DataManager:
    """
    DataManager for loading and batching the dataset.
    """

    dataset_path: str
    text: str
    tokenizer: CharTokenizer
    train_data: Tensor
    val_data: Tensor

    def __init__(
        self,
        tokenizer: CharTokenizer,
        dataset_path: str = DATASET_PATH,
    ) -> None:
        """
        Initializes the data manager by loading and processing the dataset.

        Args:
            tokenizer (CharTokenizer): The tokenizer instance to use for encoding.
            dataset_path (str): The path to the dataset file.
        """
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path

        # --- Data Loading ---
        self.load_data(self.dataset_path)

        # --- Data Splitting ---
        self.split_data(self.text)

    def load_data(self, dataset_path: str) -> None:
        """Loads the dataset from the specified file path."""
        # NOTE: This loads the entire dataset into memory
        # TODO: For larger datasets, consider streaming, chunking, or memory-mapping
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.text = f.read()

    def split_data(self, text: str, split_ratio: float = 0.9) -> None:
        """Split the data into training and validation sets."""
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

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
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

        # Create input sequences (x) and target sequences (y)
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])  # noqa E203
        y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])  # noqa E203

        # Move tensors to the configured device
        x, y = x.to(DEVICE), y.to(DEVICE)

        return x, y
