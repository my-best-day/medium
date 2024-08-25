### WIP ###

import logging
import torch

logger = logging.getLogger(__name__)


def save_checkpoint(config, model, optimizer, iter: int, val_loss: float):
    # skip checkpoint if this is not the main process
    if not config.run.is_primary:
        return

    is_wrapped = is_model_wrapped(config)

    name = "checkpoint.pt"
    checkpoint_path = config.run.checkpoints_dir / name

    torch.save(
        {
            'version': '1.2',
            'task_type': config.model.task_type,
            'iter': iter,
            'sample_counter': config.run.sample_counter,
            'model': (model.module if is_wrapped else model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config.to_dict(),
        },
        checkpoint_path
    )


def resume_from_checkpoint(config, model, optimizer, trainer):
    path = config.train.checkpoint
    if path is None:
        return

    logger.info("Resuming from checkpoint at %s", path)
    # use map_location='cpu' if GPU memory an issue (broadcasting required in that case!)
    checkpoint = torch.load(path, map_location=config.run.device)

    if str(checkpoint['version']) == '1.0':
        resume_from_checkpoint_v1(config, model, optimizer, trainer, checkpoint)
    # elif checkpoint['version'] == '2.0':
    #     resume_from_checkpoint_v2(config, model, optimizer, trainer, checkpoint)
    else:
        raise ValueError(f"Unknown checkpoint version: {checkpoint['version']}")


def resume_from_checkpoint_v1(config, model, optimizer, trainer, checkpoint):
    # torch.save(
    #     {
    #         'format': f'{config.model.task_type}.1',
    #         'version': 1.0,
    #         'iter': iter,
    #         'model': (model.module if is_wrapped else model).state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'val_loss': val_loss,
    #         'config': config.to_dict(),
    #     },
    #     checkpoint_path
    # )

    # load model state
    model_state = checkpoint['model']

    if config.model.task_type != 'mlm':
        # Filter out 'mask_lm' parameters
        model_state = {k: v for k, v in checkpoint['model'].items() if 'mask_lm' not in k}

    is_wrapped = is_model_wrapped(config)
    # in DDP, load state dict into the underlying model, otherwise, load it directly
    (model.module if is_wrapped else model).load_state_dict(model_state, strict=False)

    if config.run.parallel_mode == 'ddp':
        for param in model.module.parameters():
            torch.distributed.broadcast(param.data, src=0)

    if config.model.task_type == 'mlm':
        # load optimizer state
        optimizer_state = checkpoint['optimizer']
        optimizer.load_state_dict(optimizer_state)

        # load trainer state
        iteration = checkpoint['iter']
        val_loss = checkpoint['val_loss']
    else:
        iteration = 0
        val_loss = 0

    trainer.start_iter = iteration
    trainer.iter = trainer.start_iter

    trainer.best_val_loss = val_loss

    logger.info("Resuming from iteration %s, with val loss %s", iteration, val_loss)


def is_model_wrapped(config):
    result = config.run.parallel_mode in ('dp', 'ddp')
    return result
