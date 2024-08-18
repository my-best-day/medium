"""
Split a list of token-ids to samples of exact and fixed length (seq_len - 2)
then generate an MLM sample from it.

Works in conjunction with tokenize_text.py which convert the original text to
the list of token-ids and then store it in a binary file.

BertMlmIdsSampleGenerator creates the MLM sample by maksing 15% of the tokens.
"""
from utils.timer import Timer
import re
import logging
import gzip
import random
import msgpack
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
from transformers import BertTokenizer
from data.mlm.bert_mlm_ids_sample_generator import BertMlmIdsSampleGenerator
import itertools

logger = logging.getLogger(__name__)


def prepare_mlm_dataset(args):
    logger.info(f"Random seed: {args.seed}")

    timer = Timer("prepare_mlm_dataset")
    mlm_samples = read_and_gen_mlm_samples(args)
    logger.info(timer.step(f"read and generated {len(mlm_samples)} samples", True))

    # 80% train, 10% val, 10% test
    train_set, val_set, test_set = split_samples(mlm_samples, args)
    logger.info(timer.step(f"split samples into train: {len(train_set)}, val: {len(val_set)}, "
                f" test: {len(test_set)}", True))

    save_samples(train_set, args, "train")
    logger.info(timer.step("saved train samples", True))

    save_samples(val_set, args, "val")
    logger.info(timer.step("saved val samples", True))

    save_samples(test_set, args, "test")
    logger.info(timer.step("saved test samples", True))


def read_and_gen_mlm_samples(args):
    """
    Read tokens from the input file and generate MLM samples.
    The tokens were saved in chunks, carry remainder tokens form on chunk to the next.
    """
    tokenizer = BertTokenizer.from_pretrained(args.vocab, local_files_only=True)
    sample_generator = BertMlmIdsSampleGenerator(tokenizer, args.max_len, args.seed)

    mlm_samples = []
    with gzip.open(args.input, "rb") as inp:
        trg_len = args.max_len - 2
        reminder_tokens = []
        unpacker = msgpack.Unpacker(inp, raw=False)
        for tokens in unpacker:
            if reminder_tokens:
                tokens = reminder_tokens + tokens
                reminder_tokens = []

            for i in range(0, len(tokens), trg_len):
                sample = tokens[i:i + trg_len]
                if len(sample) == trg_len:
                    masked, label = sample_generator(sample)
                    mlm_sample = (masked, label)
                    mlm_samples.append(mlm_sample)
                else:
                    reminder_tokens = sample

    return mlm_samples


def split_samples(samples, args):
    """Split the dataset into train, validation, and test sets (80%/10%/10%)"""
    train_set, test_set = train_test_split(samples, test_size=0.2, random_state=args.seed)
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=args.seed)
    return train_set, val_set, test_set


def save_samples(samples, args, split):
    path = args.output / f"{split}_{args.label}_{args.max_len}_{args.seed}.msgpack.gz"
    # convert [(masked, label)] to [masked1, label1, masked2, label2...]
    samples = list(itertools.chain.from_iterable(samples))
    with gzip.open(path, "wb") as file:
        packed_data = msgpack.packb(samples)
        file.write(packed_data)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
Prepare MLM dataset for pre-training of a BERT model.

Example usage:
python prepare_mlm_dataset.py -i wiki.train.tokens -l wiki -o wiki/datasets
                                -m 128 -v wiki/vocab

This will generate the following files:
    wiki/datasets/train_wiki_128_123.msgpack.gz
    wiki/datasets/val_wiki_128_123.msgpack.gz
    wiki/datasets/test_wiki_128_123.msgpack.gz
where wiki is the label, 128 is the max length, and 123 is the random seed.
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="The input *token* file to be processed.")
    parser.add_argument("-l", "--label", required=True, type=str,
                        help="The label to be used for the dataset.")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="The output directory for the dataset.")
    parser.add_argument("-m", "--max-len", required=True, type=int,
                        help="The maximum length of the tokens.")
    parser.add_argument("-v", "--vocab", required=True, type=str,
                        help="The directory containing the vocabulary files.")
    parser.add_argument("-s", "--seed", type=int,
                        help="The random seed to be used.", default=None)
    args = parser.parse_args()

    args.input = Path(args.input)
    if not args.input.exists():
        raise FileNotFoundError(f"Input file {args.input} not found.")

    if not re.match(r"[\w.-]+", args.label):
        raise ValueError(f"Label must be an alphanumeric string: {args.label}")

    args.output = Path(args.output)
    if not args.output.exists():
        raise FileNotFoundError(f"Output directory {args.output} not found.")

    # validate maxlen is positive number in the rage of 16 to 2048
    if args.max_len < 16 or args.max_len > 2048:
        raise ValueError(f"Max length must be in the range of 16 to 2048: {args.max_len}")

    args.vocab = Path(args.vocab)
    if not args.vocab.exists():
        raise FileNotFoundError(f"Vocabulary directory {args.vocab} not found.")

    # if seed is not provided, generate a random seed between 0 and 1000
    if args.seed is None:
        args.seed = random.randint(0, 1000)

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = get_arguments()
    timer = Timer("prepare_mlm_dataset")
    prepare_mlm_dataset(args)
    logger.info(timer.step("Done", True))
