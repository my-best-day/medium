import gzip
import random
import pickle
import msgpack


from sklearn.model_selection import train_test_split

from transformers import BertTokenizer

from bert.dataset import BERTDataset
from bert.timer import Timer

from config import *


def preprocess_and_cache_dataset(dataset, cache_file):
    timer = Timer("preprocess_and_cache_dataset")
    preprocessed_data = []

    for i in range(len(dataset)):
        data = dataset[i]
        preprocessed_data.append(data)
    timer.print("preprocessed", True)

    # with gzip.open(cache_file, 'wb') as f:
    #     pickle.dump(preprocessed_data, f)   

    lists =[]
    for record in preprocessed_data:
        lists.append(record['bert_input'].tolist())
        lists.append(record['bert_label'].tolist())
    timer.print("converted to lists", True)

    # with gzip.open(f"{out_path}.gz", "wb") as file:
    with gzip.open(cache_file, "wb") as file:
        packed_data = msgpack.packb(lists)
        file.write(packed_data)
    timer.print("wrote msgpack.gz", True)



def _main():
    random_seed = random.randint(0, 1000)
    print(f"Random seed: {random_seed}")
    random.seed(random_seed)

    timer = Timer()
    
    ### loading all data into memory    
    corpus_movie_lines = './ignore/movie_lines.txt.gz'

    # lineId, characterId, movieId, character name, text
    with gzip.open(corpus_movie_lines, 'rb') as l:
        records = l.readlines()

    ### splitting text using special lines
    # lineId -> text
    lines = []
    for record in records:
        record = record.decode('iso-8859-1')
        objects = record.split(" +++$+++ ")
        line = objects[-1]
        lines.append(line)

    # truncate long sentences
    lines = [' '.join(line.split()[:MAX_LEN]) for line in lines]

    # Split the dataset into train, validation, and test sets (80%/10%/10%)
    train_lines, test_lines = train_test_split(lines, test_size=0.2, random_state=random_seed)
    val_lines, test_lines = train_test_split(test_lines, test_size=0.5, random_state=random_seed)

    tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)

    train_data = BERTDataset(
        train_lines, seq_len=MAX_LEN, tokenizer=tokenizer)
    val_data = BERTDataset(
        val_lines, seq_len=MAX_LEN, tokenizer=tokenizer)
    test_data = BERTDataset(
        test_lines, seq_len=MAX_LEN, tokenizer=tokenizer)
    
    preprocess_and_cache_dataset(train_data, f'./datasets/train_data_{random_seed}.msgpack.gz')
    preprocess_and_cache_dataset(val_data, f'./datasets/val_data_{random_seed}.msgpack.gz')
    preprocess_and_cache_dataset(test_data, f'./datasets/test_data_{random_seed}.msgpack.gz')
    

if __name__ == '__main__':
    for _ in range(10):
        _main()     

