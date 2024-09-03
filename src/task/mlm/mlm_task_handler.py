import torch
from torch import tensor
from torch.nn import Module
import logging
from transformer.lm.mlm.bert_mlm_model import BertMlmModel
from transformer.lm.mlm.dump_sentences import DumpSentences
from transformer.transformer import Transformer
from transformer.trainer import Trainer
from task.task_handler import TaskHandler
from task.task_handler_common import TaskHandlerCommon as THC
from task.checkpoint_utils import CheckpointUtils
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class MlmTaskHandler(TaskHandler):

    def __init__(self, config):
        self.task_type = 'mlm'
        self.config = config
        self.tokenizer = self.create_tokenizer()

        self.dumper = DumpSentences(self.tokenizer)

    def create_tokenizer(self):
        tokenizer = THC.create_bert_tokenizer(self.config)
        return tokenizer

    def create_lm_model(self):
        """
        Returns the MLM model for the given config and tokenizer.
        """
        transformer_model = THC.get_transformer_model(self.config, self.tokenizer)

        vocab_size = self.tokenizer.vocab_size
        result = BertMlmModel(transformer_model, vocab_size, apply_softmax=False)

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
        text = self.dumper.batched_debug(sentence, labels, predicted)
        logger.info("\n".join(text))

    @staticmethod
    def get_loss(logits: tensor, labels: tensor):
        loss_logits = logits.transpose(1, 2)  # Shape: [batch_size, vocab_size, seq_len]
        loss = torch.nn.functional.cross_entropy(loss_logits, labels, ignore_index=0)
        return loss

    def estimate_accuracy(self, labels: tensor, logits: tensor):
        """
        Estimate the accuracy of the model on a given task
        """
        probabilities = torch.softmax(logits, dim=-1)
        _, predicted = torch.max(probabilities, dim=-1)

        # mask: ignore padding (assumed 0) and focus on masked tokens (assumed non zero)
        flat_labels = labels.flatten()
        flat_predicted = predicted.flatten()

        mask = (flat_labels != 0)
        total = mask.sum().item()
        correct = (flat_predicted[mask] == flat_labels[mask]).sum().item()

        return total, correct

    def init_base_model_weights(self, model: Transformer):
        logger.info("Initializing base model weights")
        THC.init_transformer_model(model)

    def init_lm_head_weights(self, lm_head: torch.nn.Module):
        logger.info("Initializing LM head weights")
        torch.nn.init.normal_(lm_head.linear.weight, mean=0.0, std=0.02)
        if lm_head.linear.bias is not None:
            torch.nn.init.zeros_(lm_head.linear.bias)

        # Optional: Initialize MLM head weights with BERT's word embeddings
        # mlm_head.weight.data.copy_(bert_model.embeddings.word_embeddings.weight.data)

    def get_dataset(self, epoch, split):
        from task.mlm.bert_mlm_dataset_precached import BertMlmDatasetPrecached
        percentage = THC.get_percentage(self.config, split)
        dataset_file = THC.find_dataset_file(self.config, epoch, split)
        dataset = BertMlmDatasetPrecached(dataset_file, percentage)
        return dataset
