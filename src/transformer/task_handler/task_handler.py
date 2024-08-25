from abc import abstractmethod
from transformer.transformer import Transformer
from torch.optim.optimizer import Optimizer
from transformer.trainer import Trainer
from torch import tensor


class TaskHandler:

    @abstractmethod
    def create_lm_model(self):
        """
        Returns a model composed of the base transformer model and a language model head
        """
        raise NotImplementedError()

    @abstractmethod
    def gen_checkpoint(self, model: Transformer, optimizer: Optimizer,
                       iter: int, sample_iter: int, val_loss: float):
        """
        Generate a checkpoint for the given model, optimizer, iteration, sample count, and
        validation loss
        """
        raise NotImplementedError()

    @abstractmethod
    def resume_from_checkpoint_dict(self, model: Transformer, optimizer: Optimizer,
                                    trainer: Trainer, checkpoint: dict):
        """
        Load the base-model, language model head, optimizer state, etc. from a checkpoint
        """
        raise NotImplementedError()

    def illustrate_predictions(self, sentence: tensor, labels: tensor, predicted: tensor):
        """
        Illustrate the predictions of the model for curiosity and debugging purposes
        """
        raise NotImplementedError()

    @abstractmethod
    def get_loss(logits: tensor, labels: tensor):
        """
        Get the loss for the given logits and labels
        """
        raise NotImplementedError()

    @abstractmethod
    def estimate_accuracy(self, labels: tensor, predicted: tensor):
        """
        Estimate the accuracy of the model on a given task
        """
        raise NotImplementedError()

    @abstractmethod
    def init_base_model_weights(self, model: Transformer):
        """
        Initialize the weights of the base model
        """
        raise NotImplementedError()

    @abstractmethod
    def init_lm_head_weights(self, model: Transformer):
        """
        Initialize the weights of the language model head
        """
        raise NotImplementedError()

    @abstractmethod
    def get_dataset(self, epoch, split):
        """
        Get a dataset for the given epoch and split
        """
        raise NotImplementedError()
