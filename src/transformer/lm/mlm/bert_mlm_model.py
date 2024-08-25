import torch
from transformer.transformer import Transformer
from transformer.lm.mlm.masked_language_model import MaskedLanguageModel


class BertMlmModel(torch.nn.Module):
    """
    BERT MLM Language Model
    """

    def __init__(self, bert: Transformer, vocab_size, apply_softmax):
        """
        BERT MLM Language Model

        :param bert: BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size, apply_softmax)

    def forward(self, x):
        x = self.bert(x)
        return self.mask_lm(x)

    @property
    def base_model(self):
        """
        Provides a common interface to access the base model
        (BERT in this case).
        """
        return self.bert

    @property
    def lm_head(self):
        """
        Provides a common interface to access the language model head
        (masked LM in this case).
        """
        return self.mask_lm
