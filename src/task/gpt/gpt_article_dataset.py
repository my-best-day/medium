import logging
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GptArticleDataset(Dataset):
    """
    A GPT dataset which is fed from a files that are article aware.

    The content file contains the tokenize articles as one big list
    The indexes file contains indexes that point to samples from the content file
    adhering to article boundaries. This is to make sure the input to the model
    is a continuous logical text. that is, falls within the boundaries of a single article.

    Args:
        content_path: Path to the content file.
        indexes_path: Path to the indexes file.
        seq_len: Length of the sequences to return.
        percentage: Percentage of the dataset to use. Only used for debugging.
    """
    def __init__(self, content_path: Path, indexes_path: Path, seq_len: int,
                 percentage: float = 1.0) -> None:
        self.content_path = content_path
        self.indexes_path = indexes_path
        self.seq_len = seq_len
        self.indexes = None
        self.content = np.memmap(self.content_path, dtype='uint16', mode='r')
        self.read_indexes(self.indexes_path)

        if percentage < 1.0:
            self.indexes = self.indexes[:int(len(self.indexes) * percentage)]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a sample from the dataset.

        We use a two steps process to get the data:
        1. Read the content-index, which is called cursor here, from the indexes file.
        2. Read the data from the content file from position cursor and cursor + 1.
        3. Convert the data to tensors.

        X is a window of `seq_len` tokens starting at `cursor`.
        Y is the corresponding target window, which is X shifted one position to the right.
        The expected next token for X[0:n] is Y[n].

        Args:
            index: Index of the sample
        Returns:
            x: Input tensor
            y: Target tensor
        """
        cursor = self.indexes[index]

        xa = self.content[cursor:cursor + self.seq_len]
        ya = self.content[cursor + 1:cursor + self.seq_len + 1]

        x = torch.tensor(xa, dtype=torch.int64)
        y = torch.tensor(ya, dtype=torch.int64)

        return x, y

    def read_indexes(self, path: Path) -> list[int]:
        """
        Read indexes from a compressed numpy file.
        """
        try:
            self.indexes = np.load(path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error reading indexes from file: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error reading indexes from file: {e}")
            raise e
