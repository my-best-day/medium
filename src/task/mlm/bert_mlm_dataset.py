"""
A dataset for BERT MLM training.
"""
import random
from torch.utils.data import Dataset
import numpy as np


class BertMlmDataset(Dataset):
    """
    BertMlmDataset for preparing data for BERT model training.

    Attributes:
        tokenizer (Tokenizer): Tokenizer to convert text to tokens.
        seq_len (int): Maximum sequence length.
        corpus_lines (int): Number of lines in the corpus.
        lines (list): List of text lines.
    """

    def __init__(self, lines, tokenizer, seq_len, seed):
        """
        Initializes the BertMlmDataset.

        Args:
            lines (list): List of text lines.
            tokenizer (Tokenizer): Tokenizer to convert text to tokens.
            seq_len (int): Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.corpus_lines = len(lines)
        self.lines = lines

        # mask 15% of tokens in the text
        self.tokens_to_mask = max(1, int(0.15 * self.seq_len))
        self.rng = np.random.default_rng(seed)

        # special tokens
        self.TOKEN_ID_MASK = self.tokenizer.vocab['[MASK]']
        self.TOKEN_ID_CLS = self.tokenizer.vocab['[CLS]']
        self.TOKEN_ID_SEP = self.tokenizer.vocab['[SEP]']
        self.TOKEN_ID_PAD = self.tokenizer.vocab['[PAD]']

        # number of lines that were trimmed/padded to fit the sequence length
        self.long_count = 0
        self.long_chars = 0
        self.short_count = 0
        self.short_chars = 0

    def __len__(self):
        """number of lines in the corpus"""
        return self.corpus_lines

    def __getitem__(self, index):
        """
        Retrieves a line, masks random words, trims/pads to sequence length,
        and adds special tokens.

        Args:
            index (int): Index of the line to retrieve.

        Returns:
            dict: Dictionary containing 'bert_input' and 'bert_label' tensors.
        """
        line = self.lines[index]

        # replace random words in sentence with mask / random words
        masked, labels = self.random_word(line)

        # trim to max length leaving space for CLS, SEP, and PAD tokens
        if len(masked) > self.seq_len - 2:
            self.long_count += 1
            self.long_chars += len(masked) - (self.seq_len - 2)
            masked = masked[:self.seq_len - 2]
            labels = labels[:self.seq_len - 2]

        elif len(masked) < self.seq_len - 2:
            self.short_count += 1
            self.short_chars += (self.seq_len - 2) - len(masked)
            padding = [self.TOKEN_ID_PAD for _ in range(self.seq_len - 2 - len(masked))]
            masked.extend(padding)
            labels.extend(padding)

        # Add CLS, SEP, and PAD tokens to the start and end of sentences
        masked = [self.TOKEN_ID_CLS] + masked + [self.TOKEN_ID_SEP]
        labels = [self.TOKEN_ID_PAD] + labels + [self.TOKEN_ID_PAD]

        return masked, labels

    def random_word_text(self, text):
        # tokenize the text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.random_word_token_ids(token_ids)

    def random_word_token_ids(self, token_ids):
        """
        Masks 15% of tokens in the text.

        Args:
            text (str): Input text.

        Returns:
            tuple: Tuple containing masked tokens and labels.
        """        # list of token_ids, some are masked
        output = []
        # list of token_ids of masked tokens, 0 for non-masked tokens
        output_label = []

        masked_indices = self.rng.choice(self.seq_len, self.tokens_to_mask, replace=False)

        for i, token_id in enumerate(token_ids):
            if i in masked_indices:
                # Randomly decide the type of masking
                prob = random.random()

                # 80% chance to mask
                if prob < 0.8:
                    output_token_id = self.TOKEN_ID_MASK

                # 10% chance to replace with a random token
                elif prob < 0.9:
                    output_token_id = random.randrange(len(self.tokenizer.vocab))

                # 10% chance to leave the token as it is
                else:
                    output_token_id = token_id

                output.append(output_token_id)
                output_label.append(token_id)
            else:
                output.append(token_id)
                output_label.append(0)

        assert len(output) == len(output_label)
        return output, output_label
