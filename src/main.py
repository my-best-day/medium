import glob
import torch
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

from bert.bert import BERT
from bert.timer import Timer
from bert.trainer import BERTTrainer, BERTTrainerSingleDataset, BERTTrainerPreprocessedDatasets
from bert.bertlm import BERTLM
from bert.dataset import BERTDataset, BERTDatasetPrecached

import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# todo:
# + see how batches are used
# - add randomization in the sampling 
#   - I'm not sure how ipmortant this is, there is nothing special in the order of the sentences
# - add estimate_loss
# - see about how to move lines/data to device. gpt.py does it differently than here
#
# - optimize network structure folowing some of Andrej Karpathy's suggestions: norm before activation, etc
# - add early stopping
# - train / validation / set ?
# - try running on Aleph with larger network, batch, MAX_LEN, etc.
# - see if we can see the tokens in the output

def _main(args):
    # figured out the next run id
    run_id = get_next_run_id(Path('./logs'))
    print(f'run_id: {run_id}')

    logs_dir = Path('./logs') / f'run{run_id}'
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = Path('./checkpoints') / f'run{run_id}'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)

    bert_model = BERT(
        vocab_size=len(tokenizer.vocab),
        d_model=D_MODEL,
        n_layers=N_LAYER,
        heads=HEADS,
        dropout=DROPOUT,
        max_len=MAX_LEN
    ).to(device)

    bert_lm = BERTLM(bert_model, len(tokenizer.vocab)).to(device)

    if False:
        # train_data = BERTDatasetPrecached(
        #     './datasets/train_data_12.pkl.gz')
        # train_data = BERTDatasetPrecached(
        #     './datasets/train_data_12.pkl')
        # train_data = BERTDatasetPrecached(
        #     './datasets/train_data_28.msgpack.gz')
        # train_data = BERTDatasetPrecached(
        #     './datasets/train_data_12.msgpack')

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
            log_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
            print_every=EVAL_INTERVAL,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=20,
            tokenizer=tokenizer,
            device=device,
            dataset_dir=Path('./datasets'),
            dataset_pattern='train_data_*.msgpack'
        )

    if args.checkpoint:
        bert_trainer.load_checkpoint(args.checkpoint)

    bert_trainer.train()


def get_args():
    """
    Parses command-line arguments for the training script.

    Args:
    -c, --continue: Flag to load the checkpoint and continue training from there.
    --cp, --checkpoint <path>: Path to a specific checkpoint. The continue flag is expected.
    --run <path>: Path to a run directory in the format run<id>. The continue flag is expected.

    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Process arguments for training.")

    # Add arguments
    parser.add_argument('-c', '--continue', dest='cont', action='store_true', help='Continue training from the last checkpoint.')
    parser.add_argument('--cp', '--checkpoint', type=str, metavar='<path>', help='Path to a specific checkpoint. Requires -c/--continue.')
    parser.add_argument('--run', type=str, metavar='<path>', help='Path to a run directory in the format run<id>. Requires -c/--continue.')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs to train.')

    # Parse arguments
    args = parser.parse_args()

    # Validating that --cp and --run are used with -c/--continue
    if (args.cp or args.run) and not args.cont:
        parser.error("--cp/--checkpoint and --run require -c/--continue.")

    if args.checkpoint is not None:
        path = Path(args.checkpoint)
        if not path.exists():
            parser.error(f'Checkpoint {path} does not exist.')        

    if args.epochs <= 0:
        parser.error('Number of epochs must be positive.')

    # Validate --run argument
    if args.run is not None:
        prefix, run_id_str = args.run[:3], args.run[3:]
        if prefix != 'run' or not run_id_str.isdigit():
            parser.error("--run argument must be in the format 'run<number>'.")
        else:
            args.run_id = int(run_id_str)    

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

