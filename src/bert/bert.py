"""
Define a BERT model.
"""
import torch
import logging
from bert.encoder_layer import EncoderLayer
from bert.embedding import BERTEmbedding


class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model, n_layers, heads, dropout, seq_len):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        logging.info("BERT: vocab_size: $s, d_model: %s, n_layers: %s, heads: %s, dropout: %s",
                     vocab_size, d_model, n_layers, heads, dropout)

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.max_len = seq_len

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model,
                                       seq_len=self.max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout, seq_len) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
