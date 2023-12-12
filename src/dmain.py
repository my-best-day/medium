from pathlib import Path
import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

from bert.bert import BERT
from bert.trainer import BERTTrainerPreprocessedDatasets
from bert.bertlm import BERTLM
from bert.dataset import BERTDataset

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _main():

    tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)


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

        bert_lm = torch.nn.DataParallel(bert_lm)
        bert_lm = bert_lm.to(device)

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
        dataset_pattern='train_data_*.msgpack'
        )

    # use the first available checkpoint id
    for i in range(100):
        import glob
        files = glob.glob(f'./checkpoints/bart_{i}_*.pt')
        if len(files) == 0:
            bert_trainer.id = i

    bert_trainer.train()

if __name__ == '__main__':
    _main()

