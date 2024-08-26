import logging
from transformer.task_handler.task_handler import TaskHandler
from transformer.lm.mlm.bert_mlm_model import BertMlmModel
from transformer.lm.mlm.dump_sentences import DumpSentences
from transformer.task_handler.task_handler_common import TaskHandlerCommon as THC
from transformer.transformer import Transformer
from torch.optim.optimizer import Optimizer
from transformer.trainer import Trainer
import torch

logger = logging.getLogger(__name__)


class MlmTaskHandler(TaskHandler):

    def __init__(self, config, tokenizer):
        self.task_type = 'mlm'
        self.config = config
        self.tokenizer = tokenizer

        self.dumper = DumpSentences(tokenizer)

    def create_lm_model(self):
        """
        Returns the MLM model for the given config and tokenizer.
        """
        transformer_model = THC.get_transformer_model(self.config, self.tokenizer)

        vocab_size = len(self.tokenizer.vocab)
        result = BertMlmModel(transformer_model, vocab_size, apply_softmax=False)

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

    def illustrate_predictions(
            self, sentence: torch.tensor, labels: torch.tensor, predicted: torch.tensor):
        """
        Illustrate the predictions of the model for curiosity and debugging purposes
        """
        text = self.dumper.batched_debug(sentence, labels, predicted)
        logger.info("\n".join(text))

    @staticmethod
    def get_loss(logits: torch.tensor, labels: torch.tensor):
        loss_logits = logits.transpose(1, 2)  # Shape: [batch_size, vocab_size, seq_len]
        loss = torch.nn.functional.cross_entropy(loss_logits, labels, ignore_index=0)
        return loss

    def estimate_accuracy(self, labels: torch.tensor, predicted: torch.tensor):
        """
        Estimate the accuracy of the model on a given task
        """
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
        from data.mlm.bert_mlm_dataset_precached import BertMlmDatasetPrecached
        percentage = THC.get_percentage(self.config, split)
        dataset_file = THC.find_dataset_file(self.config, epoch, split)
        dataset = BertMlmDatasetPrecached(dataset_file, percentage)
        return dataset
