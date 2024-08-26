import torch
import logging
from transformer.task_handler.task_handler import TaskHandler
from transformer.lm.classifier.bert_classifier_model import BertClassifierModel
from data.sst2_dataset import Sst2Dataset
from transformer.task_handler.task_handler_common import TaskHandlerCommon as THC
from transformer.task_handler.checkpoint_utils import CheckpointUtils
from transformer.transformer import Transformer
from torch.optim.optimizer import Optimizer
from transformer.trainer import Trainer

logger = logging.getLogger(__name__)


class Sst2TaskHandler(TaskHandler):

    def __init__(self, config, tokenizer):
        self.task_type = 'sst2'
        self.config = config
        self.tokenizer = tokenizer

    def create_lm_model(self):
        transformer_model = THC.get_transformer_model(self.config, self.tokenizer)

        vocab_size = len(self.tokenizer.vocab)
        result = BertClassifierModel(transformer_model, vocab_size)

        return result

    def gen_checkpoint(self, model: Transformer, optimizer: Optimizer,
                       iter: int, sample_iter: int, val_loss: float):

        checkpoint = CheckpointUtils.gen_checkpoint(self.config, self.task_type, model, optimizer,
                                                    iter, sample_iter, val_loss)
        return checkpoint

    def resume_from_checkpoint_dict(self, model: Transformer, optimizer: Optimizer,
                                    trainer: Trainer, checkpoint: dict):
        CheckpointUtils.resume_from_checkpoint_dict(self.config, self.task_type, model,
                                                    optimizer, trainer, checkpoint)

    def illustrate_predictions(
            self, sentence: torch.tensor, labels: torch.tensor, predicted: torch.tensor):
        """
        Illustrate the predictions of the model for curiosity and debugging purposes
        """
        # currently no illustration for SST-2
        pass

    @staticmethod
    def get_loss(logits: torch.tensor, labels: torch.tensor):
        # labels should be [batch_size]
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def estimate_accuracy(self, labels: torch.tensor, predicted: torch.tensor):
        """
        Estimate the accuracy of the model on a given task
        """
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
        assert split in ('train', 'val')
        if split == 'train':
            prefix = 'train'
        elif split == 'val':
            prefix = 'validation'

        filename = f'{prefix}-00000-of-00001.parquet'
        path = self.config.run.datasets_dir / filename
        dataset = Sst2Dataset(path, self.tokenizer, self.config.model.seq_len)

        return dataset
