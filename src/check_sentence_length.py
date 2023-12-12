import glob
import torch
from bert.dataset import BERTDatasetPrecached
from bert.dump_sentences import describe
from bert.timer import Timer

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _main():
    ### loading all data into memory
    torch.no_grad()

    dataset_files = glob.glob('datasets/train_data_*.msgpack.gz')
    dataset_files = sorted(dataset_files)
    dataset_file = dataset_files[0]
    print(f"Loading dataset from {dataset_file}")

    dataset = BERTDatasetPrecached(dataset_file)
   

    sentence_lengths = []
    labels_lengths = []
    timer = Timer("load precached dataset")
    for i, data in enumerate(dataset):
        sentence, labels = data
        if sentence.any():
            nonzero = sentence.nonzero()
            max = nonzero.max()
            maximum = max.item() + 1
            del nonzero
            del max
        else:
            maximum = 0        
        sentence_lengths.append(maximum)
        del sentence

        if labels.any():
            nonzero = labels.nonzero()
            max = nonzero.max()
            maximum = max.item() + 1
            del nonzero
            del max
        else:
            maximum = 0
        del labels
        labels_lengths.append(maximum)

        if (i + 1) % 1000 == 0:
            timer.print(f"Processed {i + 1} samples", restart=True)

    print("done processing tensors")
    sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.float)
    labels_lengths = torch.tensor(labels_lengths, dtype=torch.float)
    describe(sentence_lengths)
    describe(labels_lengths)

if __name__ == '__main__':
    _main()

