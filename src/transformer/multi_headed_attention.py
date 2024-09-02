"""Multi-Headed Attention module for a Transformer model."""
import math
import torch


### attention layers
class MultiHeadedAttention(torch.nn.Module):
    """
    Multi-Head Attention module for a Transformer model.
    """
    def __init__(self, heads, d_model, max_len, dropout, is_gpt, use_flash):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.max_len = max_len
        self.dropout_value = dropout
        self.dropout = torch.nn.Dropout(dropout)
        self.is_causal = is_gpt
        self.flash = use_flash and torch.cuda.is_available() and \
            hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

        if not self.flash and self.is_causal:
            # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(
                torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) -->
        # (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        if self.flash:
            # need to set dropout manually when using flash attention
            dropout = self.dropout_value if self.training else 0
            context = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=self.is_causal)
        else:
            # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) -->
            #      (batch_size, h, max_len, max_len)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

            if self.is_causal and not self.flash:
                scores = scores.masked_fill(
                    self.bias[:, :, :self.max_len, :self.max_len] == 0, float('-inf'))

            # fill 0 mask with super small number so it wont affect the softmax weight
            # (batch_size, h, max_len, max_len)
            scores = scores.masked_fill(mask == 0, -1e9)

            # (batch_size, h, max_len, max_len)
            # softmax to put attention weight for all non-pad tokens
            # max_len X max_len matrix of attention
            weights = torch.nn.functional.softmax(scores, dim=-1)
            weights = self.dropout(weights)

            # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) -->
            #      (batch_size, h, max_len, d_k)
            context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) -->
        #      (batch_size, max_len, d_model)
        context = context \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        projected = self.output_linear(context)
        out = self.dropout(projected)
        return out
