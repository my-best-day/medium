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

    def forward(self, hidden_states):
        hidden_states = self.bert(hidden_states)
        logits = self.classifier_model(hidden_states)
        return logits

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
        (classifier in this case).
        """
        return self.classifier_model
