import os
import torch
import configargparse
from pathlib import Path

from utils.config import RunConfig, TrainConfig, ModelConfig, Config

class Start:
    def __init__(self, config):
        self.config = config
        self.init_mode_set_device()

    def init_mode_set_device(self):
        """
        Initialize parallel mode and device.
        Sets config.run.device
        """
        config = self.config
        parallel_mode = config.run.parallel_mode
        if parallel_mode == 'single':
            config.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif parallel_mode == 'dp':
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError('DataParallel training requires multiple GPUs.')
        elif parallel_mode == 'ddp':
            os.environ['MASTER_ADDR'] = config.run.dist_master_addr
            os.environ['MASTER_PORT'] = config.run.dist_master_port
            torch.cuda.set_device(config.run.local_rank)
            config.run.device = torch.device('cuda', torch.cuda.current_device())
            torch.distributed.init_process_group(backend=config.run.dist_backend, init_method='env://')
        else:
            raise Exception(f'Unknown parallel mode {parallel_mode}. Valid values are single, dp, ddp.')


    def train(self):
        self.tokenizer = self.get_tokenizer()

        model = self.get_model(self.tokenizer)
        self.model = self.wrap_model(model)

        total_parameters = sum([p.nelement() for p in self.model.parameters()])
        print(f"Total Parameters: {total_parameters:,}")

        self.trainer = self.get_trainer()
        if self.config.train.checkpoint is not None:
            self.trainer.load_checkpoint()

        self.trainer.train()

    def get_tokenizer(self):
        if self.config.run.case == 'movies':
            from transformers import BertTokenizer
            path = self.config.run.base_dir / 'vocab/bert-it-vocab.txt'
            path = str(path)
            result = BertTokenizer.from_pretrained(path, local_files_only=True)
        elif self.config.run.case == 'instacart':
            from instacart.instacart_tokenizer import InstacartTokenizer
            path = self.config.run.base_dir / 'vocab' / 'instacart_vocab.txt'
            result = InstacartTokenizer(path)
        else:
            raise Exception(f'Unknown case. {self.config.run.case}')
        return result

    def get_model(self, tokenizer):
        from bert.bert import BERT
        from bert.bertlm import BERTLM

        model_config = self.config.model
        vocab_size = len(tokenizer.vocab)

        bert_model = BERT(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            heads=model_config.heads,
            dropout=model_config.dropout,
            seq_len=model_config.seq_len
        )

        bert_lm = BERTLM(bert_model, vocab_size)

        return bert_lm

    def wrap_model(self, model):
        mode = self.config.run.parallel_mode
        model = model.to(self.config.run.device)
        if mode == 'single':
            pass
        elif mode == 'dp':
            model = torch.nn.DataParallel(model)
        elif mode == 'ddp':
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.config.run.local_rank])
        else:
            raise Exception(f'Unknown parallel mode {mode}. Valid values are single, dp, ddp.')
        return model

    def get_trainer(self):
        from bert.trainer import BERTTrainerPreprocessedDatasets

        trainer = BERTTrainerPreprocessedDatasets(
            self.config,
            self.model,
            tokenizer=self.tokenizer
        )
        return trainer

def get_args() -> configargparse.Namespace:
    """
    Parses command-line arguments for the training script.
    argparse.Namespace: Parsed command-line arguments.
    """
    # Create the parser
    parser = configargparse.ArgumentParser(
        description="Process arguments for training.",
        default_config_files=['./config.ini'],
    )

    # Add arguments
    # model arguments
    parser.add_argument('--seq-len', type=int, default=None, help='Sequence length')
    # parser.add_argument('--vocab-size', type=int, default=None, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=None, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=None, help='Number of layers')
    parser.add_argument('--heads', type=int, default=None, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')

    # train arguments
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--val-interval', type=int, default=None, help='Valuation interval')
    parser.add_argument('--checkpoint', '--cp', type=str, default=None, metavar='<path>', help='Path to a specific checkpoint.')
    parser.add_argument('--start_epoch', '--se', type=int, default=0, help='Epoch to start training from')
    parser.add_argument('--end-epoch', '--ee', type=int, default=19, help='Epoch to end training')
    parser.add_argument('--dataset-pattern', type=str, default=None, help='Dataset pattern')
    parser.add_argument('--val-dataset-pattern', type=str, default=None, help='Validation pattern')

    # run arguments
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory for logs and checkpoints.')
    parser.add_argument('--run-id', type=int, default=None, help='Run id. If not specified, the next available run id will be used.')
    parser.add_argument('--datasets-dir', type=str, default='datasets', help='Directory containing datasets.')

    parser.add_argument('--parallel-mode', type=str, default='single', choices=['single', 'dp', 'ddp'], help='Parallel mode for training')
    parser.add_argument('--local-rank', type=int, default=None, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    parser.add_argument('--case', type=str, default=None, choices=['movies', 'instacart'], help='Case to run')

    # add dist_master_addr, dist_master_port, dist_backend
    parser.add_argument('--dist-master-addr', type=str, default='localhost', help='Address of the master node.')
    parser.add_argument('--dist-master-port', type=str, default='12355', help='Port of the master node.')
    parser.add_argument('--dist-backend', type=str, default='nccl', help='Backend for distributed training.')

    # add dp, ddp
    parser.add_argument('--dp', action='store_true', help='Use DataParallel for training.')
    parser.add_argument('--ddp', action='store_true', help='Use DistributedDataParallel for training.')

    # Parse arguments
    args = parser.parse_args()

    if args.end_epoch < args.start_epoch:
        parser.error(f'Invalid start epoch {args.start_epoch} and end epoch {args.end_epoch}. Start epoch must be <= end epoch.')

    if args.case is None:
        parser.error('Case must be specified.')

    if args.dp:
        args.parallel_mode = 'dp'

    if args.ddp:
        args.parallel_mode = 'ddp'

    args.base_dir = Path(args.base_dir)
    if not args.base_dir.exists():
        parser.error(f'Base directory {args.base_dir} does not exist.')

    args.datasets_dir = args.base_dir / args.datasets_dir
    if not args.datasets_dir.exists():
        parser.error(f'Datasets directory {args.datasets_dir} does not exist.')

    if args.run_id is None:
        args.run_id = get_next_run_id(args.base_dir / 'runs')

    if args.parallel_mode == 'ddp':
        if args.local_rank is None:
            parser.error('Local rank must be specified when using DDP.')

        if args.local_rank < 0:
            parser.error(f'Invalid local rank {args.local_rank}. Must be >= 0.')

        if args.local_rank > torch.cuda.device_count():
            parser.error(f'Invalid local rank {args.local_rank}. Must be < number of GPUs.')

    if args.checkpoint is not None and not args.checkpoint.exists():
        parser.error(f'Checkpoint {args.checkpoint} does not exist.')

    return args

def get_next_run_id(parent_path: Path):
    """
    Iterate from 1 to 1000, check if parent_path/run{id} exists, return first id that doesn't exist. Raise exception if all ids are taken.
    """
    for i in range(1000):
        path = parent_path / f'run{i}'
        if not path.exists():
            return i
    raise Exception('All run ids are taken.')

def get_config_objects(args):
    model_config = ModelConfig(
        seq_len=args.seq_len,
        # vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        checkpoint=args.checkpoint,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        learning_rate=args.learning_rate,
        dataset_pattern=args.dataset_pattern,
        val_dataset_pattern=args.val_dataset_pattern,
    )
    run_config = RunConfig(
        base_dir = args.base_dir,
        run_id = args.run_id,
        parallel_mode = args.parallel_mode,
        local_rank = args.local_rank,
        dist_master_addr = args.dist_master_addr,
        dist_master_port = args.dist_master_port,
        dist_backend = args.dist_backend,
        case = args.case,
        datasets_dir = args.datasets_dir,
    )
    config = Config(
        model=model_config,
        train=train_config,
        run=run_config,
    )
    return config


def _main():
    args = get_args()
    config = get_config_objects(args)
    print(config)
    start = Start(config)
    start.train()


if __name__ == '__main__':
    _main()