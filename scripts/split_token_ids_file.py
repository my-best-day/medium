import re
import gzip
import random
import msgpack
import argparse
import logging
from utils.timer import Timer
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def gen_token_ids_split_files(args: argparse.Namespace) -> None:
    """
    Split a token ids file into train, validation, and test files.

    Args:
        args: The command line arguments.
    """

    token_ids = read_token_ids(args.input)

    train_set, val_set, test_set = split_token_ids(token_ids, args.seed)

    save_split_chunks(args.output, train_set, "train", args.label, args.seed)
    save_split_chunks(args.output, val_set, "val", args.label, args.seed)
    save_split_chunks(args.output, test_set, "test", args.label, args.seed)


def split_token_ids(token_ids: List[int], seed: int) -> tuple:
    """
    Split token ids into train, validation, and test sets.

    Args:
        token_ids (List[int]): A list of token IDs.
        seed (int): The seed used for randomization or chunking.
    Returns:
        tuple: A tuple containing three lists:
            - train_set (List[List[int]]): The training set.
            - val_set (List[List[int]]): The validation set.
            - test_set (List[List[int]]): The test set.
    """

    # read the token ids, split them into 40 consecutive token id chunks
    token_id_chunks = chunk_token_ids(token_ids, 40)

    # split the chunks into train, validation, and test sets
    train_set, test_set = train_test_split(token_id_chunks, test_size=0.2, random_state=seed)
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=seed)

    return train_set, val_set, test_set


def chunk_token_ids(token_ids: List[int], num_chunks: int) -> List[List[int]]:
    """
    Split token ids into multiple parts.

    Args:
        token_ids (List[int]): A list of token IDs.
        num_chunks (int): The number of chunks to split the token IDs into.
    Returns:
        List[List[int]]: A list of chunks, each chunk is a list of token IDs.
    """
    token_ids_chunks = []
    part_size = (len(token_ids) + num_chunks - 1) // num_chunks

    for i in range(num_chunks):
        start = i * part_size
        end = (i + 1) * part_size
        token_ids_chunks.append(token_ids[start:end])

    return token_ids_chunks


def save_split_chunks(
        output_dir_path: Path,
        token_id_chunks: List[List[int]],
        split: str, label: str,
        seed: int) -> None:
    """
    Save token ID chunks to a compressed MsgPack file.

    Args:
        output_dir_path (Path): The output directory path.
        token_id_chunks (List[List[int]]): A list chunks, each chunk is a list of token IDs.
        split (str): The split name (e.g., "train", "val", "test").
        label (str): A label associated with the data (e.g., dataset or model label).
        seed (int): The seed used for randomization or chunking, included in the filename.
    """
    path = output_dir_path / f"{split}_{label}_{seed}.msgpack.gz"
    packer = msgpack.Packer()
    with gzip.open(path, "wb") as out:
        for token_id_list in token_id_chunks:
            out.write(packer.pack(token_id_list))


def read_token_ids(path):
    """
    Read token ids from a file.

        Args:
        path (Path): The path to the token ids file.

    Returns:
        List[int]: A list of token IDs.
    """
    token_ids = []

    with gzip.open(path, "rb") as inp:
        unpacker = msgpack.Unpacker(inp, raw=False)
        for tokens in unpacker:
            token_ids.extend(tokens)

    return token_ids


def get_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="""Split a token ids file into train, validation, and test files.""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="The input *token ids* file to be split.")
    parser.add_argument("-l", "--label", required=True, type=str,
                        help="The label to be used for the dataset.")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="The output directory for the dataset.")
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

    # if seed is not provided, generate a random seed between 0 and 1000
    if args.seed is None:
        args.seed = random.randint(0, 1000)

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args = get_arguments()

    timer = Timer(__file__)
    gen_token_ids_split_files(args)
    logger.info(timer.step("Done", True))
