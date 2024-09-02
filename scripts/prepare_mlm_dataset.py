"""
This script prepares an MLM dataset to for training, validation, and testing
of a BERT model. .

Plan:
1. open the clean file. the clean file has the top and bottom removed and only
   contains the text of the book.

2. read the file and split the text into chunks.

for now we'll split the text by chunks of ((max-len - 2) * 4) characters, counting
on the huristic that a token is on average 4 characters long.

we may collect stats about how many chunks are too long, how many are too short.
and by how much.

3. feed the chunks into the BERT dataset to create a preprocessed dataset.
   and dump it to a file that we will use during training.
"""
from task.mlm.bert_mlm_dataset import BertMlmDataset
from utils.timer import Timer

import re
import logging
import gzip
import random
import msgpack
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from pathlib import Path
import argparse


logger = logging.getLogger(__name__)


def prepare_mlm_dataset(args):
    logger.info(f"Random seed: {args.seed}")

    with open(args.input, "r") as file:
        text = file.read()

    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s,.!?\'"-]+', '', text)

    # remove consequitive white spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # split the text into chunks
    # leave room for the [CLS] and [SEP] tokens
    # huristic: a token is on average 4 characters long
    chunk_character_count = ((args.max_len - 2) * 4)
    chunks = [text[i:i + chunk_character_count] for i in range(0, len(text), chunk_character_count)]

    # Split the dataset into train, validation, and test sets (80%/10%/10%)
    train_chunks, test_chunks = train_test_split(chunks, test_size=0.2, random_state=args.seed)
    val_chunks, test_chunks = train_test_split(test_chunks, test_size=0.5, random_state=args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.vocab, local_files_only=True)

    train_data = BertMlmDataset(train_chunks, seq_len=args.max_len, tokenizer=tokenizer)
    val_data = BertMlmDataset(val_chunks, seq_len=args.max_len, tokenizer=tokenizer)
    test_data = BertMlmDataset(test_chunks, seq_len=args.max_len, tokenizer=tokenizer)

    train_file = args.output / f"train_dk1_{args.max_len}_{args.seed}.msgpack.gz"
    val_file = args.output / f"val_dk1_{args.max_len}_{args.seed}.msgpack.gz"
    test_file = args.output / f"test_dk1_{args.max_len}_{args.seed}.msgpack.gz"

    preprocess_and_cache_dataset(train_data, train_file)
    preprocess_and_cache_dataset(val_data, val_file)
    preprocess_and_cache_dataset(test_data, test_file)


def preprocess_and_cache_dataset(dataset, cache_file):
    timer = Timer("preprocess_and_cache_dataset")

    # generate the preprocessed dataset
    # interleave the tokens and labels
    preprocessed_data = []
    for i in range(len(dataset)):
        tokens, labels = dataset[i]
        preprocessed_data.append(tokens)
        preprocessed_data.append(labels)
    timer.print("preprocessed", True)

    print(f"About max len.... items: {len(preprocessed_data) / 2}, "
          f"shorts: {dataset.short_count}, "
          f" short char avg: {dataset.short_chars / dataset.short_count:.2f}, "
          f"longs: {dataset.long_count}"
          f" long char avg: {dataset.long_chars / dataset.long_count:.2f}")

    # with gzip.open(f"{out_path}.gz", "wb") as file:
    with gzip.open(cache_file, "wb") as file:
        packed_data = msgpack.packb(preprocessed_data)
        file.write(packed_data)

    timer.print("wrote msgpack.gz", True)


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
                        help="The input text file to be processed.")
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

    if not args.label.isalnum():
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
    prepare_mlm_dataset(args)
