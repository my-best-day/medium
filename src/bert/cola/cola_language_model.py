"""
Predicting if a sentence is gramatrically correct
"""
import torch


class ColaLanguageModel(torch.nn.Module):
    """
    Predicting if a sentence is gramatrically correct
    """

    def __init__(self, hidden: int, num_labels: int):
        super(ColaLanguageModel, self).__init__()
        self.hidden = hidden
        self.classifier = torch.nn.Linear(hidden, num_labels)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            # Initialize weights using Xavier initialization (or any other suitable method)
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, hidden_states):
        # hidden_states[0] is the last hidden state (batch_size, seq_len, hidden_size)
        # We only want the [CLS] token (the first token)
        cls_output = hidden_states[0][:, 0, :]  # (batch_size, hidden_size)

        logits = self.classifier(cls_output)

        return logits
