"""Encoder layer of a Transformer model."""
import torch
from .feed_forward import FeedForward
from .multi_headed_attention import MultiHeadedAttention


class EncoderLayer(torch.nn.Module):
    """Encoder layer of a Transformer model."""
    # def __init__(self, d_model=768, heads=12, feed_forward_hidden=768 * 4, dropout=0.1):
    def __init__(self, d_model, heads, feed_forward_hidden, dropout, max_len):
        super(EncoderLayer, self).__init__()
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model, max_len=max_len)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)

        # 1. norm
        normed1 = self.layernorm1(embeddings)
        # 2. self attention
        interacted = self.self_multihead(normed1, normed1, normed1, mask)
        # 3. residual layer
        residual1 = embeddings + interacted
        # 4. norm
        normed2 = self.layernorm2(residual1)
        # 5. feed forward
        feed_forward_out = self.feed_forward(normed2)
        # 6. residual layer
        residual2 = residual1 + feed_forward_out
        return residual2
