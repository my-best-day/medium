"""
TBD
"""
import os
import logging
import multiprocessing
import torch
import warnings

from utils.args import get_args
from utils.logging import config_logging
from utils.args_to_config import get_config

from torch_configurator import TorchConfigurator

# Our checkpoint includes non tensor data, so we cannot use weights_only=True.
# This poses a security risk, but we are not concerned about it.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")


def config_wandb(config):
    import wandb
    wandb.init(
        project=config.run.case,
        config=config.to_dict(),
        name=f'run{config.run.run_id}',
        dir=config.run.run_dir,
    )


def run(config):
    task_handler = create_task_handler(config)
    configurator = TorchConfigurator(config, task_handler)
    configurator.configure()
    trainer = configurator.trainer
    if config.train.test:
        trainer.test()
    else:
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


def initialize_config():
    args = get_args()
    config = get_config(args)
    return config


def create_task_handler(config):
    from task.mlm.mlm_task_handler import MlmTaskHandler
    from task.sst2.sst2_task_handler import Sst2TaskHandler
    from task.cola.cola_task_handler import ColaTaskHandler
    from task.gpt.gpt_task_handler import GptTaskHandler
    task_type = config.model.task_type
    if task_type == 'mlm':
        task_handler = MlmTaskHandler(config)
    elif task_type == 'sst2':
        task_handler = Sst2TaskHandler(config)
    elif task_type == 'cola':
        task_handler = ColaTaskHandler(config)
    elif task_type == 'gpt':
        task_handler = GptTaskHandler(config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return task_handler


def main(config):
    logfile_path = config.run.logs_dir / 'log.txt'
    config_logging(logfile_path)

    if config.run.is_primary:
        logging.info(config.model)
        logging.info(config.train)
        logging.info(config.run)

        if config.run.wandb:
            config_wandb(config)

    if config.run.parallel_mode == 'ddp':
        run_ddp(config, config.run.nproc)
    else:
        run(config)


if __name__ == "__main__":
    config = initialize_config()
    main(config)
