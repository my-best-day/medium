from transformer.task_handler.task_handler import TaskHandler
from transformer.lm.gpt.gpt_model import GptModel
from transformer.task_handler.task_handler_common import get_transformer_model


class GptTaskHandler(TaskHandler):

    def __init__(self, config):
        self.task_type = 'gpt'
        self.config = config

    def create_lm_model(self, tokenizer):
        transformer_model = get_transformer_model(self.config, tokenizer)

        vocab_size = len(tokenizer.vocab)
        result = GptModel(transformer_model, vocab_size)

        return result
