import logging

from args import get_args
from utils.logging import config_logging
from args_to_config import get_config

import warnings
# Suppress specific warning
warnings.filterwarnings("ignore", message="\'has_cuda\' is deprecated")
warnings.filterwarnings("ignore", message="\'has_cudnn\' is deprecated")
warnings.filterwarnings("ignore", message="\'has_mps\' is deprecated")
warnings.filterwarnings("ignore", message="\'has_mkldnn\' is deprecated")


# TODO: clustered ddp, use ddp_rank and ddp_local_rank
# TODO: adjust max-iter, eval-iter, and micro-step-count based on number of gpus
# TODO: shutdown process group
# TODO: checkpoint: save and load
# TODO:

def config_wandb(config):
    import wandb
    wandb.init(
        project=config.run.case,
        config=config.to_dict(),
        name=f'run{config.run.run_id}',
        dir=config.run.run_dir,
    )

def go(config):
    from main_torch import create_objects
    trainer = create_objects(config)
    trainer.train()

def _main():
    args = get_args()
    config = get_config(args)

    logfile_path = config.run.logs_dir / 'log.txt'
    config_logging(logfile_path)

    if config.run.is_primary:
        logging.info(config.model)
        logging.info(config.train)
        logging.info(config.run)

        if config.run.wandb:
            config_wandb(config)

    go(config)

if __name__ == "__main__":
    _main()