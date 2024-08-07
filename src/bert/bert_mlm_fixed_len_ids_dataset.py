"""
A dataset for BERT MLM training.

XXX: NOT USED NOT TESTED XXX
"""
from torch.utils.data import Dataset
from bert.bert_mlm_ids_sample_generator import BertMlmIdsSampleGenerator


class BertMlmFixedLenIdsDataset(Dataset):
    """
    Bert dataset for samples which are already tokenized and exactly seq_len long.

    Attributes:
        tokenizer (Tokenizer): Tokenizer for special and random tokens
        seq_len (int): Sequence length.
        samples (list): List of samples.
        seed (int): Random seed.
    """

    def __init__(self, samples, tokenizer, seq_len, seed):
        """
        Initializes the dataset.

        Args:
            samples (list): List of samples, all of the same length.
            tokenizer (Tokenizer): Tokenizer for special and random tokens.
            seq_len (int): sequence length.
            seed (int): Random seed.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

        self.sample_generator = BertMlmIdsSampleGenerator(tokenizer, seq_len, seed)

    def __len__(self):
        """number of samples in the corpus"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieves a line, masks random words, trims/pads to sequence length,
        and adds special tokens.

        Args:
            index (int): Index of the line to retrieve.

        Returns:
            dict: Dictionary containing 'bert_input' and 'bert_label' tensors.
        """
        sample = self.samples[index]

        # MLM masking
        masked, labels = self.sample_generator(sample)

        return masked, labels
