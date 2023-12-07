from pathlib import Path
import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

from bert.bert import BERT
from bert.trainer2 import BERTTrainer2
from bert.bertlm import BERTLM
from bert.dataset import BERTDataset

PREPARE_DATA = False
MAX_LEN = 64 # 64
BATCH_SIZE = 64
EVAL_INTERVAL = 200
EVAL_ITERS = 50
D_MODEL = 768
N_LAYER = 2
HEADS = 12
DROPOUT = 0.1
LEARNING_RATE = 5e-4

PROFILE = "xxbee"
if PROFILE == "bee":
    PREPARE_DATA = False
    MAX_LEN = 16 # 32 
    BATCH_SIZE = 32
    EVAL_INTERVAL = 200
    EVAL_ITERS = 20
    D_MODEL = 48 # 96 # 192
    HEADS = 4
    N_LAYER = 1
else:
    PREPARE_DATA = False
    MAX_LEN = 64
    BATCH_SIZE = 128
    EVAL_INTERVAL = 200
    EVAL_ITERS = 20
    D_MODEL = 768 # 768 
    N_LAYER = 4
    HEADS = 12

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

### loading all data into memory
corpus_movie_lines = './datasets/movie_lines.txt'

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


tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)


'''test run'''
train_data = BERTDataset(
   lines, seq_len=MAX_LEN, tokenizer=tokenizer)

train_loader = DataLoader(
   train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

eval_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

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


bert_trainer = BERTTrainer2(
    bert_lm, 
    train_data,
    log_dir=Path('./logs'),
    checkpoint_dir=Path('./checkpoints'),
    print_every=EVAL_INTERVAL,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=20,
    device=device,
    tokenizer=tokenizer)

bert_trainer.train()
