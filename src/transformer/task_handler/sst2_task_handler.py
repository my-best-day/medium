from transformer.task_handler.task_handler import TaskHandler
from transformer.lm.classifier.bert_classifier_model import BertClassifierModel
from transformer.task_handler.task_handler_common import get_transformer_model


class Sst2TaskHandler(TaskHandler):

    def __init__(self, config):
        self.task_type = 'sst2'
        self.config = config

    def create_lm_model(self, tokenizer):
        transformer_model = get_transformer_model(self.config, tokenizer)

        vocab_size = len(tokenizer.vocab)
        result = BertClassifierModel(transformer_model, vocab_size, apply_softmax=False)

        return result
