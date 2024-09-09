import torch
from transformer.transformer import Transformer
from .generative_language_model import GenerativeModel


class GptModel(torch.nn.Module):
    """
    GPT Language Model
    """

    def __init__(self, gpt: Transformer, vocab_size: int):
        """
        """
        super().__init__()
        self.gpt = gpt
        self.gen_lm = GenerativeModel(gpt.d_model, vocab_size)

    def forward(self, x):
        # (b, seq_len) -> (b, seq_len, d_model)
        x = self.gpt(x)
        # (b, seq_len, d_model) -> (b, seq_len, vocab_size)
        return self.gen_lm(x)

    @property
    def base_model(self):
        """
        Provides a common interface to access the base model
        (GPT in this case).
        """
        return self.gpt

    @property
    def lm_head(self):
        """
        Provides a common interface to access the language model head
        (Generative in this case).
        """
        return self.gen_lm
