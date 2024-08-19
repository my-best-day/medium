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
        self.lm_head = GenerativeModel(gpt.d_model, vocab_size)

    def forward(self, x):
        x = self.gpt(x)
        return self.lm_head(x)
