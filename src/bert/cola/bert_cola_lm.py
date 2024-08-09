import torch
from bert.bert import BERT
from bert.cola.cola_language_model import ColaLanguageModel


class BertColaLanguageModel(torch.nn.Module):
    """
    BERT Cola Language Model
    """

    def __init__(self, bert: BERT):
        """
        BERT Cola Language Model
        """
        super().__init__()
        self.bert = bert
        self.cola_lm = ColaLanguageModel(self.bert.d_model, 2)

    def forward(self, hiden_states):
        hiden_states = self.bert(hiden_states)
        logits = self.cola_lm(hiden_states)
        return logits
