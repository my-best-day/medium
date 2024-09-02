from transformers import PreTrainedTokenizer

END_OF_TEXT = "<|endoftext|>"


class MockGptTokenizer(PreTrainedTokenizer):
    def __init__(self):
        self.vocab = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, END_OF_TEXT: 5}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.special_tokens = [END_OF_TEXT]
        # super's __init__ uses self.vocab, so we need to call it after setting it up
        super().__init__()

    def _tokenize(self, text):
        return list(text)  # Tokenize by character

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[END_OF_TEXT])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, END_OF_TEXT)

    def get_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens):
        # we don't add special tokens
        assert add_special_tokens is False, "add_special_tokens must be False"
        tokens = self._tokenize(text)
        return [self._convert_token_to_id(token) for token in tokens]

    def decode(self, token_ids):
        return ''.join([self._convert_id_to_token(token_id) for token_id in token_ids])

    @property
    def vocab_size(self):
        return len(self.vocab)
