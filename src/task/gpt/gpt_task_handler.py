import torch
from torch import tensor
from torch.nn import Module
import logging
from transformer.lm.gpt.gpt_model import GptModel
from transformer.lm.gpt.generate_sentence import GenerateSentence
from transformer.transformer import Transformer
from transformer.trainer import Trainer
from task.task_handler import TaskHandler
from task.task_handler_common import TaskHandlerCommon as THC
from task.checkpoint_utils import CheckpointUtils
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class GptTaskHandler(TaskHandler):

    def __init__(self, config):
        self.task_type = 'gpt'
        self.config = config
        self.tokenizer = self.create_tokenizer()

        seq_len = self.config.model.seq_len
        max_new_tokens = int(0.33 * seq_len)
        self.dumper = GenerateSentence(self.tokenizer, seq_len, max_new_tokens)

    def create_tokenizer(self):
        tokenizer = THC.create_gpt_tokenizer(self.config)
        return tokenizer

    def create_lm_model(self):
        transformer_model = THC.create_transformer_model(self.config, self.tokenizer)

        vocab_size = self.tokenizer.vocab_size
        result = GptModel(transformer_model, vocab_size)

        return result

    def gen_checkpoint(self, model: Transformer, optimizer: Optimizer,
                       iter: int, sample_iter: int, val_loss: float, lr: float):

        checkpoint = CheckpointUtils.gen_checkpoint(self.config, self.task_type, model, optimizer,
                                                    iter, sample_iter, val_loss, lr)
        return checkpoint

    def resume_from_checkpoint_dict(self, model: Transformer, optimizer: Optimizer,
                                    trainer: Trainer, checkpoint: dict):

        CheckpointUtils.resume_from_checkpoint_dict(self.config, self.task_type, model,
                                                    optimizer, trainer, checkpoint)

    def illustrate_predictions(self, model: Module,
                               sentence: tensor, labels: tensor, predicted: tensor):
        """
        Illustrate the predictions of the model for curiosity and debugging purposes
        """
        text = self.dumper.batched_debug(model, sentence, labels, predicted)
        logging.info("\n".join(text))

    @staticmethod
    def get_loss(logits: tensor, labels: tensor):
        loss_logits = logits.transpose(1, 2)  # Shape: [batch_size, vocab_size, seq_len]
        loss = torch.nn.functional.cross_entropy(loss_logits, labels, ignore_index=0)
        return loss

    def estimate_accuracy(self, labels: tensor, logits: tensor):
        """
        There is no accuracy for GPT task
        """
        return 0, 0

    def init_base_model_weights(self, model: Transformer):
        logger.info("Initializing base model weights")
        THC.init_transformer_model(model)

    def init_lm_head_weights(self, lm_head: torch.nn.Module):
        logger.info("Initializing LM head weights")
        torch.nn.init.normal_(lm_head.classifier.weight, mean=0.0, std=0.02)
        if lm_head.classifier.bias is not None:
            torch.nn.init.zeros_(lm_head.classifier.bias)

    def get_dataset(self, epoch, split):
        from task.gpt.gpt_token_ids_dataset import GptTokenIdsDataset
        seq_len = self.config.model.seq_len
        percentage = THC.get_percentage(self.config, split)
        dataset_file = THC.find_dataset_file(self.config, epoch, split)
        dataset = GptTokenIdsDataset(dataset_file, seq_len, percentage)
        return dataset
