"""
Generate MLM samples by masking 15% of the tokens.

Assuming sample of identical, seq-len, length.
"""
import random
import numpy as np


class BertMlmIdsSampleGenerator:
    """
    BertMlmIdsSampleGenerator for generating MLM samples by masking tokens.

    Attributes:
        tokenizer (Tokenizer): Tokenizer to convert text to tokens.
        seq_len (int): Maximum sequence length.
        seed (int): Seed for random number generation.
        tokens_to_mask (int): Number of tokens to mask in the text.
        rng (Generator): Random number generator.

        TOKEN_ID_MASK (int): Token ID for the '[MASK]' token.
        TOKEN_ID_CLS (int): Token ID for the '[CLS]' token.
        TOKEN_ID_SEP (int): Token ID for the '[SEP]' token.
        TOKEN_ID_PAD (int): Token ID for the '[PAD]' token.
    """

    def __init__(self, tokenizer, seq_len, seed):
        """
        Initializes the BertMlmIdsSampleGenerator.

        Args:
            lines (list): List of text lines.
            tokenizer (Tokenizer): Tokenizer to convert text to tokens.
            seq_len (int): Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

        # mask 15% of tokens in the text
        self.tokens_to_mask = max(1, int(0.15 * self.seq_len))
        self.rng = np.random.default_rng(seed)

        # special tokens
        self.TOKEN_ID_MASK = self.tokenizer.vocab['[MASK]']
        self.TOKEN_ID_CLS = self.tokenizer.vocab['[CLS]']
        self.TOKEN_ID_SEP = self.tokenizer.vocab['[SEP]']
        self.TOKEN_ID_PAD = self.tokenizer.vocab['[PAD]']

    def __call__(self, token_ids):
        """
        Retrieves a line, masks random words, trims/pads to sequence length,
        and adds special tokens.

        Args:
            index (int): Index of the line to retrieve.

        Returns:
            dict: Dictionary containing 'bert_input' and 'bert_label' tensors.
        """
        # replace random words in sentence with mask / random words
        masked, labels = self.mask_tokens(token_ids)

        # Add CLS, SEP, and PAD tokens to the start and end of sentences
        masked = [self.TOKEN_ID_CLS] + masked + [self.TOKEN_ID_SEP]
        labels = [self.TOKEN_ID_PAD] + labels + [self.TOKEN_ID_PAD]
        return masked, labels

    def mask_tokens(self, raw_sample):
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

        for i, token_id in enumerate(raw_sample):
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
