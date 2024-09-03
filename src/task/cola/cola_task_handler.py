import torch
from torch import tensor
from torch.nn import Module
import logging
from transformer.lm.classifier.bert_classifier_model import BertClassifierModel
from transformer.transformer import Transformer
from transformer.trainer import Trainer
from task.task_handler import TaskHandler
from task.cola.cola_dataset import ColaDataset
from task.task_handler_common import TaskHandlerCommon as THC
from task.checkpoint_utils import CheckpointUtils
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class ColaTaskHandler(TaskHandler):

    def __init__(self, config):
        self.task_type = 'cola'
        self.config = config
        self.tokenizer = self.create_tokenizer()

    def create_tokenizer(self):
        tokenizer = THC.create_bert_tokenizer(self.config)
        return tokenizer

    def create_lm_model(self):
        transformer_model = THC.create_transformer_model(self.config, self.tokenizer)

        vocab_size = self.tokenizer.vocab_size
        result = BertClassifierModel(transformer_model, vocab_size)

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
        # currently no illustration for CoLA
        pass

    @staticmethod
    def get_loss(logits: tensor, labels: tensor):
        # labels should be [batch_size]
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def estimate_accuracy(self, labels: tensor, logits: tensor):
        """
        Estimate the accuracy of the model on a given task
        """
        probabilities = torch.softmax(logits, dim=-1)
        _, predicted = torch.max(probabilities, dim=-1)

        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return total, correct

    def init_base_model_weights(self, model: Transformer):
        logger.info("Initializing base model weights")
        THC.init_transformer_model(model)

    def init_lm_head_weights(self, lm_head: torch.nn.Module):
        logger.info("Initializing LM head weights")
        torch.nn.init.normal_(lm_head.classifier.weight, mean=0.0, std=0.02)
        if lm_head.classifier.bias is not None:
            torch.nn.init.zeros_(lm_head.classifier.bias)

    def get_dataset(self, epoch, split):
        assert split in ('train', 'val', 'test')
        if split == 'train':
            filename = 'in_domain_train.tsv'
        elif split == 'val':
            filename = 'in_domain_dev.tsv'
        elif split == 'test':
            filename = 'out_of_domain_dev.tsv'
        path = self.config.run.datasets_dir / filename
        dataset = ColaDataset(path, self.tokenizer, self.config.model.seq_len)
        return dataset
