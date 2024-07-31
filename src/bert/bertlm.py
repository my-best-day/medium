import torch
from bert.bert import BERT
from bert.masked_language_model import MaskedLanguageModel


class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    """

    def __init__(self, bert: BERT, vocab_size, apply_softmax):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size, apply_softmax)

    def forward(self, x):
        x = self.bert(x)
        return self.mask_lm(x)
