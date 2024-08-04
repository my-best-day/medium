"""
This script prepares the Oliver Twist dataset for training, evaluation, and testing
a BERT model with MLM.

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
from bert.bert_dataset import BERTDataset
from bert.timer import Timer

import re
import logging
import gzip
import random
import msgpack
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer

from pathlib import Path


logger = logging.getLogger(__name__)

MAX_LEN = 128
INPUT_PATH = Path("./ignore/dickens/dickens/dickens.txt")
LABEL = "dickens1"
CASE_DIR = Path("dickens")


def preprocess_and_cache_dataset(dataset, cache_file):
    timer = Timer("preprocess_and_cache_dataset")

    # generate the preprocessed dataset
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


def _main():
    random_seed = random.randint(0, 1000)
    logger.info(f"Random seed: {random_seed}")

    with open(INPUT_PATH, "r") as file:
        text = file.read()

    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s,.!?\'"-]+', '', text)

    # remove consequitive white spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # split the text into chunks
    # leave room for the [CLS] and [SEP] tokens
    # huristic: a token is on average 4 characters long
    chunk_character_count = ((MAX_LEN - 2) * 4)
    chunks = [text[i:i + chunk_character_count] for i in range(0, len(text), chunk_character_count)]

    # Split the dataset into train, validation, and test sets (80%/10%/10%)
    train_chunks, test_chunks = train_test_split(chunks, test_size=0.2, random_state=random_seed)
    val_chunks, test_chunks = train_test_split(test_chunks, test_size=0.5, random_state=random_seed)

    vocab_dir = CASE_DIR / "vocab"
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, local_files_only=True)

    train_data = BERTDataset(train_chunks, seq_len=MAX_LEN, tokenizer=tokenizer)
    val_data = BERTDataset(val_chunks, seq_len=MAX_LEN, tokenizer=tokenizer)
    test_data = BERTDataset(test_chunks, seq_len=MAX_LEN, tokenizer=tokenizer)

    datasets_path = CASE_DIR / "datasets"

    train_file = datasets_path / f"train_dk1_{MAX_LEN}_{random_seed}.msgpack.gz"
    val_file = datasets_path / f"val_dk1_{MAX_LEN}_{random_seed}.msgpack.gz"
    test_file = datasets_path / f"test_dk1_{MAX_LEN}_{random_seed}.msgpack.gz"

    preprocess_and_cache_dataset(train_data, train_file)
    preprocess_and_cache_dataset(val_data, val_file)
    preprocess_and_cache_dataset(test_data, test_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _main()
