"""
Define a Transformer (BERT/GPT) model.
"""
import torch
import logging
from transformer.encoder_layer import EncoderLayer
from transformer.embedding import ModelEmbedding


class Transformer(torch.nn.Module):
    """
    Transformer model: Can act as both BERT and GPT.
    """

    def __init__(self, vocab_size, d_model, n_layers, heads, dropout, seq_len, is_gpt, use_flash):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        :param seq_len: max sequence length
        :param is_gpt: flag to use GPT or BERT
        :param use_flash: flag to use flash attention
        """
        logging.info("MODEL: vocab_size: %s, d_model: %s, n_layers: %s, heads: %s, dropout: %s",
                     vocab_size, d_model, n_layers, heads, dropout)

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.max_len = seq_len

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding, sum of positional, token embeddings
        self.embedding = ModelEmbedding(vocab_size=vocab_size, embed_size=d_model,
                                        seq_len=self.max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout, seq_len, is_gpt, use_flash)
                for _ in range(n_layers)])

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
