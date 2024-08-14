"""
TBD
"""
import os
import logging
import multiprocessing
import torch

from utils.args import get_args
from utils.logging import config_logging
from utils.args_to_config import get_config

from torch_main import create_objects_and_trainer

# Suppress specific warning
# warnings.filterwarnings("ignore", message="\'has_cuda\' is deprecated")
# warnings.filterwarnings("ignore", message="\'has_cudnn\' is deprecated")
# warnings.filterwarnings("ignore", message="\'has_mps\' is deprecated")
# warnings.filterwarnings("ignore", message="\'has_mkldnn\' is deprecated")


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


def run(config):
    trainer = create_objects_and_trainer(config)
    trainer.train()


def run_ddp(config, world_size):

    processes = []
    for local_rank in range(world_size):
        process = multiprocessing.Process(target=ddp_worker, args=(config, local_rank, world_size))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()


def ddp_worker(config, local_rank, world_size):

    config.run.local_rank = local_rank
    config.run.is_primary = local_rank == 0
    os.environ['MASTER_ADDR'] = config.run.dist_master_addr
    os.environ['MASTER_PORT'] = config.run.dist_master_port
    os.environ['LOCAL_RANK'] = str(local_rank)  # probably not needed
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.distributed.init_process_group(
        backend=config.run.dist_backend,
        world_size=world_size,
        rank=local_rank,
    )

    run(config)

    torch.distributed.destroy_process_group()


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

    if config.run.parallel_mode == 'ddp':
        run_ddp(config, args.nproc)
    else:
        run(config)


if __name__ == "__main__":
    _main()
