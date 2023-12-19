import torch
import argparse
from pathlib import Path
from transformers import BertTokenizer

from bert.bert import BERT
from bert.bertlm import BERTLM
from bert.trainer import BERTTrainerSingleDataset, BERTTrainerPreprocessedDatasets


from config import *
from instacart.instacart_tokenizer import InstacartTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _main(args, base_dir: Path):
    # figured out the next run id
    base_logs_dir = base_dir / 'logs'
    base_checkpoints_dir = base_dir / 'checkpoints'

    run_id = get_next_run_id(base_logs_dir)
    print(f'run_id: {run_id}')

    logs_dir = base_logs_dir / f'run{run_id}'
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = base_checkpoints_dir / f'run{run_id}'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if False:
        tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)
    else:
        tokenizer = InstacartTokenizer()

    bert_model = BERT(
        vocab_size=len(tokenizer.vocab),
        d_model=D_MODEL,
        n_layers=N_LAYER,
        heads=HEADS,
        dropout=DROPOUT,
        max_len=MAX_LEN
    )

    bert_lm = BERTLM(bert_model, len(tokenizer.vocab))

    if torch.cuda.is_available():
        # bert_model = torch.nn.DataParallel(bert_model)
        bert_model = bert_model.to(device)
        bert_lm = bert_lm.to(device)

        if args.data_parallel:
            # bert_lm = torch.nn.DataParallel(bert_lm)
            bert_lm = torch.nn.parallel.DistributedDataParallel(bert_lm)


    if True:
        ZZZZZZ

        bert_trainer = BERTTrainerSingleDataset(
            bert_lm,
            log_dir=Path('./logs'),
            checkpoint_dir=Path('./checkpoints'),
            print_every=EVAL_INTERVAL,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=20,
            tokenizer=tokenizer,
            device=device,
            dataset=train_data
        )
    else:
        bert_trainer = BERTTrainerPreprocessedDatasets(
            bert_lm,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
            print_every=EVAL_INTERVAL,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=args.epochs,
            tokenizer=tokenizer,
            device=device,
            dataset_dir=Path('./datasets32'),
            dataset_pattern='train_data_*.msgpack',
            eval_pattern='val_data_*.msgpack',
        )

    if args.checkpoint:
        bert_trainer.load_checkpoint(args.checkpoint)

    bert_trainer.train()


def get_args():
    """
    Parses command-line arguments for the training script.

    Args:
    --cp, --checkpoint <path>: Path to a specific checkpoint. The continue flag is expected.
    -e, --epochs <int>: Number of epochs to train.

    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Process arguments for training.")

    # Add arguments
    parser.add_argument('--checkpoint', '--cp', type=str, default=None, metavar='<path>', help='Path to a specific checkpoint.')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--data-parallel', '-d', action='store_true', help='Use DataParallel for training')

    # Parse arguments
    args = parser.parse_args()

    if args.checkpoint is not None:
        path = Path(args.checkpoint)
        if not path.exists():
            parser.error(f'Checkpoint {path} does not exist.')

    if args.epochs <= 0:
        parser.error('Number of epochs must be positive.')

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

if __name__ == '__main__':
    args = get_args()
    _main(args)
