"""
Generative model
"""
import torch


class GenerativeModel(torch.nn.Module):
    """
    Generative model
    """

    def __init__(self, hidden: int, num_labels: int):
        super(GenerativeModel, self).__init__()
        self.classifier = torch.nn.Linear(hidden, num_labels)

    def forward(self, hidden_state):
        # hidden_states is the last hidden state (batch_size, seq_len, hidden_size)
        logits = self.classifier(hidden_state)
        return logits
