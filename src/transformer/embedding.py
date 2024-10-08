import torch
import logging

from .positional_embedding import PositionalEmbedding


class ModelEmbedding(torch.nn.Module):
    """
    Model Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of ModelEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len, dropout):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, sequence):
        try:
            x = self.token(sequence)
            y = self.position(sequence)
            z = x + y
            return self.dropout(z)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in ModelEmbedding forward: {e}")
            raise e
