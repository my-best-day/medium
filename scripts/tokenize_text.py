"""
Convert a text file to token ids which can later be used to create a dataset for
pre-training a BERT model.

This conversion takes time and is identical when we create multiple datasets.
This tool lets us do the text to token-ids convertion once and then reuse it
for multiple datasets.

We optionally parallelize the tokanization which further saves time. We work on
chunks of the text to avoid memory issues.

The next step of the process is to create the pre-cached dataset which is done
by the script `prepare_mlm_fixed_len_dataset_from_ids.py`.
"""
import re
import gzip
import argparse
import joblib
from transformers import BertTokenizer
from bert.timer import Timer
import msgpack
import itertools
import logging

logger = logging.getLogger(__name__)


def chop_trailing_sub_word(text):
    """
    Chops the last word if it's not complete - it will be prepend to the next chunk.
    """
    last_white_char = find_last_complete_word_position(text)
    if last_white_char > 0:
        remainder = text[last_white_char:]
        text = text[:last_white_char]
    else:
        remainder = ""
    return text, remainder


def find_last_complete_word_position(text):
    """
    Search from the end backword for the fisrt white space character the precedes
    non-white space characters. This is the last (sub)word.
    """
    match = re.search(r'\s+[^\s]$', text)
    if match:
        last_white_char = match.start()
    else:
        last_white_char = -1
    return last_white_char


def tokenize(vocab, text):
    """convert text to token ids. helps when we parallelize the tokenization"""
    tokenizer = BertTokenizer.from_pretrained(vocab, local_files_only=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens


def tokenize_file(args):
    """
    Tokenize a text file and write the tokenized samples to a binary file.
    """
    packer = msgpack.Packer()
    remainder_chars = ""

    # break the input file to chunks to avoid memory issues
    chunk_size = args.chunk_size * 1024 * 1024

    with gzip.open(args.output, 'wb') as out:
        with open(args.input, "r", encoding='utf-8') as inp:
            chunk_count = 0
            while True:
                chunk_count += 1
                loop_timer = Timer("tokenize loop")
                steps_timer = Timer("tokenize steps")

                text = inp.read(chunk_size)
                logger.debug(steps_timer.step("read chunk", True))

                if not text:
                    # we may lose the remainder chars, that's acceptable
                    return
                logger.debug(steps_timer.step("chunk size: ", len(text)))

                # remove special characters, clean spaces if needed
                text = re.sub(r'[^a-zA-Z0-9\s,.!?\'"-]+', '', text)
                # normalize white spaces if the tokenizer doesn't do this for us
                if args.clean_spaces:
                    text = re.sub(r"\s+", " ", text)
                logger.debug(steps_timer.step("clean text", True))

                # prepend the remainder characters from the previous chunk
                if remainder_chars:
                    text = remainder_chars + text
                logger.debug(steps_timer.step("prepend remainder", True))

                # chop the last word if it's not complete
                text, remainder_chars = chop_trailing_sub_word(text)
                logger.debug(steps_timer.step("chop trailing", True))

                # tokenize the text, pralllelize if nproc is greater than 1
                if args.nproc > 1:
                    with joblib.parallel_backend('loky', n_jobs=args.nproc):
                        text_chunks = divide_text(text, args.nproc)
                        tokens_list = joblib.Parallel()(
                            joblib.delayed(tokenize)(args.vocab, chunk) for chunk in text_chunks
                        )
                        tokens = list(itertools.chain.from_iterable(tokens_list))
                else:
                    tokens = tokenize(args.vocab, text)
                logger.debug(steps_timer.step("tokenize", True))

                # write the samples to the output file
                out.write(packer.pack(tokens))
                logger.debug(steps_timer.step("write", True))
                logger.info(loop_timer.step(f"chunk {chunk_count} done", True))


def divide_text(text, chunk_count):
    """
    Breaks the text to chunks of approximately equal size for parallel processing.
    """
    chunks = []
    start = 0
    text_length = len(text)
    chunk_size = (text_length + chunk_count - 1) // chunk_count

    while start < text_length:
        end = start + chunk_size
        end = min(end, text_length)

        # Move back to the end of the last complete word
        while end < text_length and not text[end].isspace():
            end -= 1

        chunks.append(text[start:end])

        start = end
        # Skip whitespace at the start of the next chunk
        while start < text_length and text[start].isspace():
            start += 1

    return chunks


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a text file to tokens which can later "
        "be used to create a dataset for pre-training a BERT model."
    )
    parser.add_argument("-i", "--input", type=str, help="The input file.")
    parser.add_argument("-o", "--output", type=str, help="The output file.")
    parser.add_argument("-v", "--vocab", type=str, help="The dir containing the vocab files.")
    parser.add_argument("-c", "--chunk-size", type=int, default=20, help="The chunk size in MB.")
    parser.add_argument("-n", "--nproc", type=int, default=1, help="The number of processes.")
    parser.add_argument("--clean-spaces", type=int, choices=[0, 1], default=-1,
                        help="Clean spaces: 1 for True, 0 for False")

    args = parser.parse_args()
    return args


def post_process_args(args, tokenizer):
    """
    convert the clean_spaces argument to a boolean.
    -1 means auto.
    """
    if args.clean_spaces == -1:
        args.clean_spaces = check_should_clean_spaces(tokenizer)
    else:
        args.clean_spaces = args.clean_spaces == 1


def check_should_clean_spaces(tokenizer):
    """
    check if the tokenizer normalize spaces.
    if it doesn't, we should clean spaces. going overboard for fun.
    """
    t1 = tokenizer.encode("h  w")
    t2 = tokenizer.encode("h w")
    should_clean_spaces = t1 != t2
    return should_clean_spaces


def _main():
    args = get_arguments()
    tokenizer = BertTokenizer.from_pretrained(args.vocab, local_files_only=True)
    post_process_args(args, tokenizer)
    tokenize_file(args)


if __name__ == "__main__":
    timer = Timer("tokenize")
    _main()
    logger.info(timer.step("done", True))
