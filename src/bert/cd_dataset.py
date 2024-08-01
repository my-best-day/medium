import re
import gzip
from torch.utils.data import Dataset

# Constants to define the lines to skip at the top and the pattern for the bottom line
SKIP_TOP_LINES = 100
BOTTOM_LINE_PATTERN = re.compile("END OF TH(E|IS) PROJECT GUTENBERG", re.IGNORECASE)

class CDDataset(Dataset):
    """
    A PyTorch Dataset for reading text files, tokenizing them, and creating sequences of a given length.
    """
    def __init__(self, path, tokenizer, seq_len):
        """
        Initialize the dataset with a file path, a tokenizer, and a sequence length.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # Tokenize the text and calculate the number of sequences
        self.tokens = self.get_tokens(path, tokenizer)
        self.corpus_lines = self.calculate_length()

    def calculate_length(self):
        """
        Calculate the number of sequences in the dataset.
        """
        return len(self.tokens) - self.seq_len + 1

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return self.corpus_lines

    def __getitem__(self, index):
        """
        Return the sequence at the given index.
        """
        return self.tokens[index:index + self.seq_len]

    @staticmethod
    def get_tokens(path, tokenizer):
        """
        Read the text file, skip the top and bottom lines, tokenize the text, and return the tokens.
        """
        with gzip.open(path, 'rt', encoding='iso-8859-1') as l:
            lines = l.readlines()
        lines = lines[SKIP_TOP_LINES:]
        for i, record in enumerate(lines):
            if BOTTOM_LINE_PATTERN.match(record):
                lines = lines[:i]
                break
        text = ' '.join(lines)
        tokens = tokenizer.tokenize(text)
        return tokens
