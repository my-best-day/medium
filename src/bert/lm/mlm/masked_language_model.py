import torch


class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size, apply_softmax):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.normlayer = torch.nn.LayerNorm(hidden)
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1) if apply_softmax else torch.nn.Identity()

    def forward(self, x):
        x = self.normlayer(x)
        x = self.linear(x)
        # might be noop of apply_softmax is False
        x = self.softmax(x)
        return x
