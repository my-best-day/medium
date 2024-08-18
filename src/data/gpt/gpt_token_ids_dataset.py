import gzip
import torch
import msgpack
from torch.utils.data import Dataset


class GptTokenIdsDataset(Dataset):
    """
    A GPT dataset, based on a list of token ids
    """
    def __init__(self, path, seq_len, percentage):
        self.path = path
        self.seq_len = seq_len

        self.token_ids = read_token_ids(path)
        if percentage < 1.0:
            self.token_ids = self.token_ids[:int(len(self.token_ids) * percentage)]

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        x = torch.tensor(self.token_ids[index:index + self.seq_len], dtype=torch.int64)
        y = torch.tensor(self.token_ids[index + 1:index + self.seq_len + 1], dtype=torch.int64)
        return x, y


def read_token_ids(path):
    """
    Read token ids from a file.
    """
    token_ids = []

    with gzip.open(path, "rb") as inp:
        unpacker = msgpack.Unpacker(inp, raw=False)
        for tokens in unpacker:
            token_ids.extend(tokens)

    return token_ids
