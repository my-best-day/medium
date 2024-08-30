### WIP ###
import logging
from transformer.task_handler.task_handler_common import TaskHandlerCommon as THC
from transformer.transformer import Transformer
from torch.optim.optimizer import Optimizer
from transformer.trainer import Trainer

logger = logging.getLogger(__name__)


class CheckpointUtils:

    @staticmethod
    def gen_checkpoint(config, task_type: str, model: Transformer, optimizer: Optimizer,
                       iter: int, sample_iter: int, val_loss: float):
        # skip checkpoint if this is not the main process
        if not config.run.is_primary:
            return

        the_model = THC.unwrap_model(model)

        checkpoint = {
            'version': '1.2',
            'task_type': task_type,
            'iter': iter,
            'sample_iter': sample_iter,

            'base_model': the_model.base_model.state_dict(),
            'lm_head': the_model.lm_head.state_dict(),

            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config.to_dict(),
        }

        return checkpoint

    @staticmethod
    def resume_from_checkpoint_dict(config, task_type: str, model: Transformer,
                                    optimizer: Optimizer, trainer: Trainer, checkpoint: dict):

        if checkpoint['version'] != '1.2':
            raise ValueError(f"Unknown checkpoint version: {checkpoint['version']}")

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
        is_resume_training = checkpoint['task_type'] == task_type and \
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
            trainer.iters = trainer.start_iter

            trainer.best_val_loss = checkpoint['val_loss']

        # log sample count, iteration, and best validation loss
        logger.info("Resuming from sample %s, iteration %s, with val loss %s",
                    trainer.sample_iter, trainer.iters, trainer.best_val_loss)
