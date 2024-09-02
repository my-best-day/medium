"""
Dataset for pre-cached BERT inputs and labels.
"""
import gzip
import torch
import msgpack
from torch.utils.data import Dataset


class BertMlmDatasetPrecached(Dataset):
    """
    Dataset for pre-cached BERT MLM inputs and labels.

    Masking is already done in the cached data.
    Cached data is a list of pairs of items. Both items are lists of token ids.
    The first item is the input sentence with a masked token at a random position.
    The second item is the label, which has the token id 0 at non-masked positions
    and zeros otherwise.
    See bert_mlm_ids_sample_generator.py for more details.
    """
    def __init__(self, path, percentage=1.0):
        self._type = None
        self._percentage = percentage
        self.cached_data = self.load_data(path)

    def load_data(self, path):
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rb') as f:
            packed_data = f.read()
            cached_data = msgpack.unpackb(packed_data, raw=False)
        if self._percentage < 1.0:
            cached_data = self.cached_data[:int(len(self.cached_data) * self._percentage)]
        return cached_data

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        result = len(self.cached_data) // 2
        if self._percentage < 1.0:
            result = int(result * self._percentage)
        return result

    def __getitem__(self, index):
        """
        Retrieves a pre-cached BERT input and label.
        """
        l_sentence = self.cached_data[index * 2]
        l_labels = self.cached_data[index * 2 + 1]
        sentence = torch.tensor(l_sentence, dtype=torch.int64)
        labels = torch.tensor(l_labels, dtype=torch.int64)
        del l_sentence
        del l_labels
        return sentence, labels
