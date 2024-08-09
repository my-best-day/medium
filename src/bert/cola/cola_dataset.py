"""
A dataset for CoLA dataset.

CoLA: The Corpus of Linguistic Acceptability consists of sentences labeled as
grammatically correct or incorrect.
"""
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from transformers import PreTrainedTokenizer


class ColaDataset(Dataset):
    """
    A dataset for CoLA dataset.
    """
    def __init__(self, path: Path, tokenizer: PreTrainedTokenizer, seq_len: int):
        """
        Initializes the CoLA dataset.
        Args:
            path (Path): Path to the CoLA file.
            tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            seq_len (int): Maximum sequence length.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.df = self.read_cola_file(path)

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves a line, masks random words, trims/pads to sequence length,
        and adds special tokens.
        """
        row = self.df.iloc[index]
        text = row['text']
        label = row['class']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        token_ids = encoding['input_ids'].squeeze()
        return token_ids, label

    @staticmethod
    def read_cola_file(path: Path):
        """
        Reads the CoLA file and returns the lines.
        """
        df = pd.read_csv(path, sep='\t', header=None, names=("src", "class", "ignore", "text"))
        return df
