import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PretokenizedDataset(Dataset):
    """
    A memory-efficient PyTorch Dataset for loading pre-tokenized data.
    It uses numpy's memory-mapping to avoid loading the entire dataset into RAM,
    which is crucial for large datasets.
    """

    def __init__(self, data_path: str, block_size: int):
        super().__init__()
        # Memory-map the file. Data is not loaded into RAM.
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        # The number of possible starting positions for a sequence
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Grab a chunk of (block_size + 1) tokens
        chunk = self.data[idx : idx + self.block_size + 1]  # noqa E203

        # Convert to int64 tensors, which is standard for embedding layers
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def create_dataloader(
    data_path: str,
    block_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Creates a PyTorch DataLoader for a pre-tokenized dataset."""
    dataset = PretokenizedDataset(data_path=data_path, block_size=block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
