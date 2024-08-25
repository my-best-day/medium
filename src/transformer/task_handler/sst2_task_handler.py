import torch
import logging
from transformer.task_handler.task_handler import TaskHandler
from transformer.lm.classifier.bert_classifier_model import BertClassifierModel
from data.sst2_dataset import Sst2Dataset
from transformer.task_handler.task_handler_common import TaskHandlerCommon as THC
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
        # skip checkpoint if this is not the main process
        if not self.config.run.is_primary:
            return

        the_model = THC.unwrap_model(model)

        checkpoint = {
            'version': '1.2',
            'task_type': self.task_type,
            'iter': iter,
            'sample_iter': sample_iter,

            'base_model': the_model.bert.state_dict(),
            'lm_head': the_model.lm_head.state_dict(),

            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict(),
        }

        return checkpoint

    def resume_from_checkpoint_dict(self, model: Transformer, optimizer: Optimizer,
                                    trainer: Trainer, checkpoint: dict):

        if checkpoint['version'] != '1.2':
            raise ValueError(f"Unknown checkpoint version: {checkpoint['version']}")

        config = self.config

        # in DDP, load state dict into the underlying model, otherwise, load it directly
        unwrapped_model = THC.unwrap_model(model)

        base_model = unwrapped_model.base_model
        base_model.load_state_dict(checkpoint['base_model'])
        config.run.init_base_model_weights = False

        # Resume training: Load the language model head, optimizer state, and trainer state
        # if the task type matches and we're not switching training modes.
        # Use this flag when continuing the same task but starting a new training phase,
        # such as switching from pre-training to fine-tuning, or when modifying the training
        # setup (e.g., learning rate schedule, dataset).
        is_resume_training = checkpoint['task_type'] == self.task_type and \
            not config.train.switch_training

        if is_resume_training:
            lm_head = unwrapped_model.lm_head
            lm_head.load_state_dict(checkpoint['lm_head'])
            config.run.init_lm_head_weights = False

            optimizer.load_state_dict(checkpoint['optimizer'])
            config.run.skip_sync_optimizer_state = False
            # back compatibility - OK to remove in the near near future
            if 'sample_count' in checkpoint:
                trainer.sample_iter = checkpoint['sample_count']
            else:
                trainer.sample_iter = checkpoint['sample_iter']
            trainer.sample_iter_start = trainer.sample_iter
            trainer.start_iter = checkpoint['iter']
            trainer.iter = trainer.start_iter

            trainer.best_val_loss = checkpoint['val_loss']

        # log sample count, iteration, and best validation loss
        logger.info("Resuming from sample %s, iteration %s, with val loss %s",
                    trainer.sample_iter, trainer.iter, trainer.best_val_loss)

        # if config.run.parallel_mode == 'ddp':  NOSONAR
        #     for param in model.module.parameters():
        #         torch.distributed.broadcast(param.data, src=0)

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
