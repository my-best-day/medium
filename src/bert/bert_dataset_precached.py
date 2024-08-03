"""
Dataset for pre-cached BERT inputs and labels.
"""
import re
import gzip
import torch
import pickle
import msgpack
from torch.utils.data import Dataset


class BERTDatasetPrecached(Dataset):
    """
    Dataset for pre-cached BERT inputs and labels.

    Masking is already done in the cached data.
    """
    def __init__(self, path, percentage=1.0):
        self._type = None
        self._percentage = percentage
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rb') as f:
            if re.search(r'\.pkl(\.gz)?$', path):
                self._type = 0
                self.cached_data = pickle.load(f)
            elif re.search(r'\.msgpack(\.gz)?$', path):
                self._type = 1
                packed_data = f.read()
                self.cached_data = msgpack.unpackb(packed_data, raw=False)
            else:
                raise ValueError(f"Unknown file type: {path}")
        if percentage < 1.0:
            self.cached_data = self.cached_data[:int(len(self.cached_data) * percentage)]
        # logging.info(timer.step(f"loaded precached dataset {path}"))

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        if self._type == 0:
            result = len(self.cached_data)
        elif self._type == 1:
            result = len(self.cached_data) // 2
        else:
            raise ValueError(f"Unknown type: {self._type}")
        if self._percentage < 1.0:
            result = int(result * self._percentage)
        return result

    def __getitem__(self, index):
        """
        Retrieves a pre-cached BERT input and label.
        """
        if self._type == 0:
            return self.cached_data[index]['bert_input'], self.cached_data[index]['bert_label']
        elif self._type == 1:
            l_sentence = self.cached_data[index * 2]
            l_labels = self.cached_data[index * 2 + 1]
            sentence = torch.tensor(l_sentence, dtype=torch.int64)
            labels = torch.tensor(l_labels, dtype=torch.int64)
            del l_sentence
            del l_labels
            return sentence, labels
        else:
            raise ValueError(f"Unknown type: {self._type}")
