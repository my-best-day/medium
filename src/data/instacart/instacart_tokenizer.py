"""
Tokenizer for the Instacart dataset.
"""
import collections


class InstacartTokenizer:
    """
    The Instacart vocab file is a list of product ids sorted by frequency.
    Dataset is a list of lists of product ids.
    """
    def __init__(self, vocab_file, do_lower_case=True):
        with open(vocab_file, "r", encoding="utf-8") as reader:
            self.vocab = reader.readlines()
        self.vocab = load_vocab(vocab_file)

    def tokenize(self, text):
        """
        Convert string to tokens.
        Not applicable in our case where the input is a list of token ids
        """
        raise NotImplementedError('tokenize is not implemented')

    def convert_tokens_to_ids(self, tokens):
        """
        Skipping, our input is a list of token ids
        """
        raise NotImplementedError('convert_tokens_to_ids is not implemented')

    @property
    def token_id_to_product_name(self):
        dictionary = getattr(self, '_token_id_to_prdocut_name', None)
        if dictionary is None:
            self._token_id_to_prdocut_name = {v: k for k, v in self.vocab.items()}
        return self._token_id_to_prdocut_name

    def convert_ids_to_tokens(self, ids):
        result = [self.token_id_to_product_name[token_id] for token_id in ids]
        return result

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError('convert_tokens_to_string is not implemented')

    @property
    def vocab_size(self):
        return len(self.vocab)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab
