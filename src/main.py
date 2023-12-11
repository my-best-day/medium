from pathlib import Path
import torch
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

def _main():
    ### loading all data into memory
    corpus_movie_lines = './datasets/movie_lines.txt'



    tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)


    '''test run'''
    if False:
        # lineId, characterId, movieId, character name, text
        with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
            records = l.readlines()

        ### splitting text using special lines
        # lineId -> text
        lines = []
        for record in records:
            objects = record.split(" +++$+++ ")
            line = objects[-1]
            lines.append(line)

        # truncate long sentences
        lines = [' '.join(line.split()[:MAX_LEN]) for line in lines]

        train_data = BERTDataset(
            lines, seq_len=MAX_LEN, tokenizer=tokenizer)
    else:        
        # train_data = BERTDatasetPrecached(
        #     './datasets/train_data_12.pkl.gz')
        # train_data = BERTDatasetPrecached(
        #     './datasets/train_data_12.pkl')
        train_data = BERTDatasetPrecached(
            './datasets/train_data_12.msgpack.gz')
        # train_data = BERTDatasetPrecached(
        #     './datasets/train_data_12.msgpack')
        pass

    # train_loader = DataLoader(
    # train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # eval_loader = DataLoader(
    #     train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    bert_model = BERT(
        vocab_size=len(tokenizer.vocab),
        d_model=D_MODEL,
        n_layers=N_LAYER,
        heads=HEADS,
        dropout=DROPOUT,
        max_len=MAX_LEN
    ).to(device)

    bert_lm = BERTLM(bert_model, len(tokenizer.vocab)).to(device)

    if True:
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
            log_dir=Path('./logs'),
            checkpoint_dir=Path('./checkpoints'),
            print_every=EVAL_INTERVAL,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=20,
            tokenizer=tokenizer,
            device=device,
            dataset_dir=Path('./datasets'),
            dataset_pattern='train_data_*.pkl'
        )

    bert_trainer.train()

if __name__ == '__main__':
    _main()

