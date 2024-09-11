import re
from pathlib import Path
import argparse
import logging
import numpy as np
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


class WikitextGptDatasetGenerator:
    ARTICLE_HEADER_PATTERN = r'^\s*=\s+[^=].*[^=]\s+=\s*$'
    ARTICLE_HEADER_REGEX = re.compile(ARTICLE_HEADER_PATTERN)

    def __init__(self, args):
        self.args = args
        self.article_tokens = []
        self.indexes = []
        self.tokenizer = self.create_tokenizer(args.vocab)

    def generate(self):
        self.process_file()
        if len(self.article_tokens) > 0:
            self.save_dataset()

    def process_file(self):
        text_file = self.args.input
        article = ""
        article_count = 0
        with open(text_file, "r") as f:
            for line in f:
                if self.is_article_header(line):
                    if len(article) > 0:
                        self.process_article(article)
                        article_count += 1
                        if article_count % 1000 == 0:
                            logger.info("Processed %s articles, %s tokens, %s indexes",
                                        article_count, len(self.article_tokens), len(self.indexes))
                    article = ""
                article += line

    @staticmethod
    def is_article_header(line):
        return WikitextGptDatasetGenerator.ARTICLE_HEADER_REGEX.match(line) is not None

    def process_article(self, article):
        clean_article = self.clean_text(article)
        # tokenize the article
        tokens = self.tokenizer.encode(clean_article, add_special_tokens=False)

        # there is an index for each token in the article which has at least
        # (seq_len + 1) tokens after it. i..seq-len will be the input, i+1..seq_len+1
        # will be the target
        seq_len = self.args.max_len
        end_index = len(tokens) - seq_len - 1
        if end_index < 0:
            return

        # add indexes to the indexes
        current_index = len(self.article_tokens)
        indexes = [current_index + i for i in range(0, end_index)]
        self.indexes.extend(indexes)

        # add tokens to the article
        self.article_tokens.extend(tokens)

    @staticmethod
    def clean_text(text):
        # remove special characters, clean spaces if needed
        text = re.sub(r'[^a-zA-Z0-9\s,.!?\'"-]+', '', text)
        # normalize white spaces if the tokenizer doesn't do this for us
        if args.clean_spaces:
            text = re.sub(r"\s+", " ", text)
        return text

    def save_dataset(self):
        self.save_content()
        self.save_indexes()

    def save_content(self):
        args = self.args
        path = args.output / f"{args.split}_{args.label}_{args.max_len}_content.npy"
        typed_tokens = np.array(self.article_tokens, dtype=np.uint16)
        np.save(path, typed_tokens)

    def save_indexes(self):
        args = self.args
        path = args.output / f"{args.split}_{args.label}_{args.max_len}_indexes.npy"
        typed_indexes = np.array(self.indexes, dtype=np.uint32)
        np.save(path, typed_indexes)

    @staticmethod
    def create_tokenizer(vocab_path):
        path = str(vocab_path)
        tokenizer = GPT2Tokenizer.from_pretrained(path, local_files_only=True)
        return tokenizer


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
Prepare GPT dataset for pre-training (/val/test)  of a GPT model.
Adhering to article boundaries.

Example usage:
python prepare_wikitext_gpt_dataset.py -i wiki.train.tokens -s train -l wiki -o wiki/datasets
                                -m 128 -v wiki/gpt_vocab

This will generate the following files:
    wiki/datasets/train_wiki_128_123.msgpack.gz
    wiki/datasets/val_wiki_128_123.msgpack.gz
    wiki/datasets/test_wiki_128_123.msgpack.gz
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="The input *token* file to be processed.")
    parser.add_argument("-s", "--split", required=True, type=str,
                        help="The split to be used for the dataset.")
    parser.add_argument("-l", "--label", required=True, type=str,
                        help="The label to be used for the dataset.")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="The output directory for the dataset.")
    parser.add_argument("-m", "--max-len", required=True, type=int,
                        help="The maximum length of the tokens.")
    parser.add_argument("-v", "--vocab", required=True, type=str,
                        help="The directory containing the vocabulary files.")
    parser.add_argument("-c", "--clean-spaces", action="store_true",
                        help="Clean spaces in the text (if the tokenizer doesn't do this).")
    args = parser.parse_args()

    args.input = Path(args.input)
    if not args.input.exists():
        raise FileNotFoundError(f"Input file {args.input} not found.")

    if not re.match(r"[\w.-]+", args.split):
        raise ValueError(f"Split must be an alphanumeric string: {args.split}")

    if not re.match(r"[\w.-]+", args.label):
        raise ValueError(f"Label must be an alphanumeric string: {args.label}")

    args.output = Path(args.output)
    if not args.output.exists():
        raise FileNotFoundError(f"Output directory {args.output} not found.")

    # validate maxlen is positive number in the rage of 16 to 2048
    if args.max_len < 12 or args.max_len > 2048:
        raise ValueError(f"Max length must be in the range of 16 to 2048: {args.max_len}")

    args.vocab = Path(args.vocab)
    if not args.vocab.exists():
        raise FileNotFoundError(f"Vocabulary directory {args.vocab} not found.")

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_arguments()
    creator = WikitextGptDatasetGenerator(args)
    creator.generate()
