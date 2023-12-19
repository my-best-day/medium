import ast
import torch
import random
import pandas as pd
from torch.utils.data import Dataset

class InstacartDataset(Dataset):
    def __init__(self, orders, vocab, seq_len):
        """
        orders is a list of lists of token id
        """
        self.vocab = vocab
        self.seq_len = seq_len
        self.size = len(orders)
        self.orders = orders

        if seq_len < 5:
            raise ValueError('seq_len should be greater than 5')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        l_sentence, l_label = self.get_plain(index)
        sentence = torch.tensor(l_sentence, dtype=torch.int64)
        labels = torch.tensor(l_labels, dtype=torch.int64)
        del l_sentence
        del l_labels
        return sentence, labels

    def get_plain(self, index):
        order = self.orders[index]

        sentence, label = self.random_token_ids(order)

        sentence = [self.vocab['[CLS]']] + sentence + [self.vocab['[SEP]']]
        label = [self.vocab['[PAD]']] + label + [self.vocab['[PAD]']]

        # Pad the sentence
        padding = [self.vocab['[PAD]']] * (self.seq_len - len(sentence))
        sentence.extend(padding)
        label.extend(padding)

        return sentence, label

    def random_token_ids(self, token_ids):
        output_label = []
        output = []

        token_ids = token_ids[:self.seq_len-2]

        # Calculate number of tokens to mask (15% of tokens in the sentence)
        num_to_mask = int(len(token_ids) * 0.15)

        # Shuffle indices to randomly select tokens to mask
        indices = list(range(len(token_ids)))
        random.shuffle(indices)
        masked_indices = set(indices[:num_to_mask])

        for i, token_id in enumerate(token_ids):
            # Token ID without cls and sep tokens
            if i in masked_indices:
                # Randomly decide the type of masking
                prob = random.random()

                # 80% chance to mask
                if prob < 0.8:
                    output.append(self.vocab['[MASK]'])

                # 10% chance to replace with a random token
                elif prob < 0.9:
                    output.append(random.randrange(len(self.vocab)))

                # 10% chance to leave the token as it is
                else:
                    output.append(token_id)

                output_label.append(token_id)
            else:
                output.append(token_id)
                output_label.append(0)

        assert len(output) == len(output_label)
        return output, output_label
