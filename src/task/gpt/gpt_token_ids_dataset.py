import logging
import gzip
import torch
import msgpack
from torch.utils.data import Dataset
from pathlib import Path

logger = logging.getLogger(__name__)


class GptTokenIdsDataset(Dataset):
    """
    A GPT dataset which is fed from a file container token ids,
    that is, the text has been tokenized and converted to token ids.

    Args:
        path: Path to the file containing the token ids.
        seq_len: Length of the sequences to return.
        percentage: Percentage of the dataset to use. Only used for debugging.
    """
    def __init__(self, path: Path, seq_len: int, percentage: float = 1.0) -> None:
        if 0.0 < percentage > 1.0:
            raise ValueError("Percentage must be between 0.0 and 1.0.")

        self.path = path
        self.seq_len = seq_len

        self.token_ids = self.read_token_ids(path)

        if percentage < 1.0:
            self.token_ids = self.token_ids[:int(len(self.token_ids) * percentage)]

    def __len__(self) -> int:
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a sample from the dataset.

        X is a window of `seq_len` tokens starting at `index`.
        Y is the corresponding target window, which is X shifted one position to the right.
        The expected next token for X[0:n] is Y[n].

        Args:
            index: Index of the sample
        Returns:
            x: Input tensor
            y: Target tensor
        """
        x = torch.tensor(self.token_ids[index:index + self.seq_len], dtype=torch.int64)
        y = torch.tensor(self.token_ids[index + 1:index + self.seq_len + 1], dtype=torch.int64)
        return x, y

    @staticmethod
    def read_token_ids(path: Path) -> list[int]:
        """
        Read token ids from a file.
        """
        token_ids = []

        try:
            with gzip.open(path, "rb") as inp:
                unpacker = msgpack.Unpacker(inp, raw=False)
                for tokens in unpacker:
                    token_ids.extend(tokens)
        except (FileNotFoundError, gzip.BadGzipFile) as e:
            logger.error(f"Error reading token ids from file: {e}")
            raise e

        return token_ids
