"""
Arguments to be used in the training script.
"""
import os
from pathlib import Path
import configargparse
import torch


def get_args() -> configargparse.Namespace:  # NOSONAR: complex function, the alternative is worse
    """
    Parses command-line arguments for the training script.
    argparse.Namespace: Parsed command-line arguments.
    """
    # Create the parser
    parser = configargparse.ArgumentParser(
        description="Process arguments for training.",
        default_config_files=['./config.ini', './local_config.ini'],
    )

    add_model_arguments(parser)
    add_train_arguments(parser)
    add_run_arguments(parser)

    # Parse arguments
    args = parser.parse_args()

    if args.end_epoch < args.start_epoch:
        parser.exit(f'Invalid start epoch {args.start_epoch} and end epoch '
                    f'{args.end_epoch}. Start epoch must be <= end epoch.')

    if args.case is None:
        parser.exit('Case must be specified.')

    if args.dp:
        args.parallel_mode = 'dp'

    if args.ddp:
        args.parallel_mode = 'ddp'

    if args.base_dir is None:
        args.base_dir = Path(args.case)
    else:
        args.base_dir = Path(args.base_dir)
    if not args.base_dir.exists():
        parser.exit(f'Base directory {args.base_dir} does not exist.')

    args.datasets_dir = args.base_dir / args.datasets_dir
    if not args.datasets_dir.exists():
        parser.exit(f'Datasets directory {args.datasets_dir} does not exist.')

    if args.run_id is None:
        args.run_id = get_next_run_id(args.base_dir / 'runs')

    if args.local_rank is None and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    # NOSONAR
    # if args.parallel_mode == 'ddp':
    #     if args.local_rank is None and 'LOCAL_RANK' in os.environ:
    #         args.local_rank = int(os.environ['LOCAL_RANK'])
    #     if args.local_rank is None:
    #         parser.error('Local rank must be specified when using DDP.')

        if args.local_rank < 0:
            parser.exit(f'Invalid local rank {args.local_rank}. Must be >= 0.')

        if args.local_rank > torch.cuda.device_count():
            parser.exit(f'Invalid local rank {args.local_rank}. Must be < number of GPUs.')

    if args.lr_scheduler == 'none':
        args.lr_scheduler = None

    args.weight_decay = float(args.weight_decay)

    # verify percentage is between 0 and 1
    if args.dataset_percentage < 0 or args.dataset_percentage > 1:
        parser.exit(f'Invalid dataset percentage {args.dataset_percentage}. '
                    'Must be between 0 and 1.')
    if args.val_dataset_percentage < 0 or args.val_dataset_percentage > 1:
        parser.exit(f'Invalid val dataset percentage {args.val_dataset_percentage}. '
                    'Must be between 0 and 1.')

    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)
        if not args.checkpoint.exists():
            parser.exit(f'Checkpoint {args.checkpoint} does not exist.')

    return args


def add_model_arguments(parser: configargparse.ArgumentParser):
    parser.add_argument('--seq-len', type=int, default=None, help='Sequence length')
    parser.add_argument('--d-model', type=int, default=None, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=None, help='Number of layers')
    parser.add_argument('--heads', type=int, default=None, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout')


def add_train_arguments(parser: configargparse.ArgumentParser):
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--val-interval', type=int, default=None, help='Valuation interval')
    parser.add_argument('--checkpoint', '--cp', type=str, default=None, metavar='<path>',
                        help='Path to a specific checkpoint.')
    parser.add_argument('--start_epoch', '--se', type=int, default=0,
                        help='Epoch to start training from')
    # ee deprecated
    parser.add_argument('--end-epoch', '--ee', type=int, default=19, help='Epoch to end training')
    parser.add_argument('--dataset-pattern', type=str, default=None, help='Dataset pattern')
    parser.add_argument('--val-dataset-pattern', type=str, default=None, help='Validation pattern')
    parser.add_argument('--max-checkpoints', type=int, default=5,
                        help='Maximum number of checkpoints to keep.')

    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--min-learning-rate', type=float,
                        default=None, help='Minimum learning rate')
    parser.add_argument('--warmup-iters', type=int, default=None, help='Warmup iterations')
    parser.add_argument('--lr-decay-iters', type=int, default=None,
                        help='Learning rate decay iterations')
    parser.add_argument('--max-iters', type=int, default=None, help='Maximum iterations')

    parser.add_argument('--lr-scheduler', type=str, default=None, help='Learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')

    parser.add_argument('--dataset-percentage', type=float, default=1.0,
                        help='Percentage of dataset to use')
    parser.add_argument('--val-dataset-percentage', type=float, default=1.0,
                        help='Percentage of validation dataset to use')


def add_run_arguments(parser: configargparse.ArgumentParser):
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory for logs and checkpoints. Defaulted to case name.')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run id. If not specified, the next available run id will be used.')
    parser.add_argument('--datasets-dir', type=str, default='datasets',
                        help='Directory containing datasets.')

    parser.add_argument('--parallel-mode', type=str, default='single',
                        choices=['single', 'dp', 'ddp'], help='Parallel mode for training')
    parser.add_argument('--local-rank', type=int, default=None,
                        help='Local rank. Necessary for using the torch.distributed.launch.')
    parser.add_argument('--case', type=str, default=None,
                        choices=['movies', 'instacart', 'dickens'], help='Case to run')
    parser.add_argument('--nproc', type=int, default=1,
                        help='Number of processes for distributed training.')

    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Use wandb for logging.')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Compile the model.')
    parser.add_argument('--async-to-device', action='store_true', default=False,
                        help='async tensors to device.')
    parser.add_argument('--fused-adamw', action='store_true', default=False,
                        help='Use fused adamw.')
    parser.add_argument('--flash', action='store_true', default=False,
                        help='Use flash attention.')

    # add dist_master_addr, dist_master_port, dist_backend
    parser.add_argument('--dist-master-addr', type=str, default='localhost',
                        help='Address of the master node.')
    parser.add_argument('--dist-master-port', type=str, default='12355',
                        help='Port of the master node.')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='Backend for distributed training.')

    # add dp, ddp
    parser.add_argument('--dp', action='store_true', help='Use DataParallel for training.')
    parser.add_argument('--ddp', action='store_true',
                        help='Use DistributedDataParallel for training.')


def get_next_run_id(parent_path: Path):
    """
    Iterate from 1 to 1000, check if parent_path/run{id} exists, return
    first id that doesn't exist. Raise exception if all ids are taken.
    """
    for i in range(1000):
        path = parent_path / f'run{i}'
        if not path.exists():
            return i
    raise RuntimeError('All run ids are taken.')
