"""
A dataset for the Stanford Sentiment Treebank (SST-2) dataset.
"""
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from transformers import PreTrainedTokenizer


class Sst2Dataset(Dataset):
    """
    A dataset for the Stanford Sentiment Treebank (SST-2) dataset.
    """

    def __init__(self, path: Path, tokenizer: PreTrainedTokenizer, sql_len: int):
        """
        Args:
            path: The path to the dataset.
            tokenizer: The tokenizer to use to tokenize the data.
            max_length: The maximum length of the tokenized text.
        """
        self.path = path
        self.tokenizer = tokenizer
        self.sql_len = sql_len

        # Load the data
        assert path.exists(), f'Error: {path} does not exist.'
        self.df = pd.read_parquet(path)

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """Returns the tokenized text and label for the given index."""
        row = self.df.iloc[index]
        text = row['sentence']
        label = row['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.sql_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        token_ids = encoding['input_ids'].squeeze()
        return token_ids, label
