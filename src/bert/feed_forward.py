import torch


class FeedForward(torch.nn.Module):
    """
    Implements a feed-forward neural network, AKA a multi-layer perceptron <3.
    """

    def __init__(self, d_model, middle_dim, dropout):
        """
        Initializes the feed-forward neural network.

        :param d_model: the number of expected features in the input (required).
        :param middle_dim: the size of the intermediate layer (required).
        :param dropout: the dropout value (required).
        """
        super(FeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
