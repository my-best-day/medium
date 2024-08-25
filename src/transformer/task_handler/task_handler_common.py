import torch
import logging
from transformer.transformer import Transformer


logger = logging.getLogger(__name__)


class TaskHandlerCommon:
    @staticmethod
    def get_transformer_model(config, tokenizer):
        """
        Returns the base, transformer model for the given config and tokenizer.
        """
        from transformer.transformer import Transformer

        model_config = config.model
        vocab_size = len(tokenizer.vocab)

        transformer_model = Transformer(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            heads=model_config.heads,
            dropout=config.train.dropout,
            seq_len=model_config.seq_len,
            is_gpt=config.model.task_type == 'gpt',
            use_flash=config.run.flash
        )

        return transformer_model

    @staticmethod
    def unwrap_model(model):
        if TaskHandlerCommon.is_model_wrapped(model):
            result = model.module
        else:
            result = model
        return result

    @staticmethod
    def is_model_wrapped(model):
        return isinstance(model,
                          (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))

    @staticmethod
    def init_transformer_model(model: Transformer):
        def init(module):
            if isinstance(module, torch.nn.Linear):
                return torch.nn.init.xavier_uniform_(module.weight)

        model.apply(init)

    @staticmethod
    def init_lm_head_weights(lm_head: torch.nn.Module):
        # standard initialization for LM head
        lm_head.weight.data.normal_(mean=0.0, std=0.02)
        lm_head.bias.data.zero_()

    @staticmethod
    def find_dataset_file(config, epoch, split):
        """
        Find the dataset file for the given epoch and split.
        """
        import glob
        if split == 'train':
            pattern = config.train.dataset_pattern
        elif split == 'val':
            pattern = config.train.val_dataset_pattern
        else:
            raise ValueError(f"Unknown split: {split}")

        pattern = str(config.run.datasets_dir / pattern)
        # add an optional .gz extension to the pattern
        dataset_files = glob.glob(pattern) + glob.glob(pattern + '.gz')
        if len(dataset_files) == 0:
            raise ValueError(f"Dataset files not found with pattern {pattern}")
        dataset_files = sorted(dataset_files)
        dataset_file = dataset_files[epoch % len(dataset_files)]

        logger.info(f"* Epoch: {epoch} - Loading dataset from {dataset_file}")

        return dataset_file

    @staticmethod
    def get_percentage(config, split):
        if split == 'train':
            percentage = config.train.dataset_percentage
        elif split == 'val':
            percentage = config.train.val_dataset_percentage
        else:
            raise ValueError(f"Unknown split: {split}")
        return percentage
