import torch
from transformer.transformer import Transformer
from .classifier_language_model import ClassifierModel


class BertClassifierModel(torch.nn.Module):
    """
    BERT Classifier Model
    """

    def __init__(self, bert: Transformer, num_labels: int):
        """
        BERT Classifier Model
        """
        super().__init__()
        self.bert = bert
        self.classifier_model = ClassifierModel(self.bert.d_model, num_labels)

    def forward(self, hiden_states):
        hiden_states = self.bert(hiden_states)
        logits = self.classifier_model(hiden_states)
        return logits
