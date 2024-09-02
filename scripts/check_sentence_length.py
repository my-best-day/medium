import glob
import torch
from task.mlm.bert_mlm_dataset_precached import BertMlmDatasetPrecached
from transformer.lm.mlm.dump_sentences import describe
from utils.timer import Timer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _main():
    ### loading all data into memory
    torch.no_grad()

    dataset_files = glob.glob('datasets/train_data_*.msgpack.gz')
    dataset_files = sorted(dataset_files)
    dataset_file = dataset_files[0]
    print(f"Loading dataset from {dataset_file}")

    dataset = BertMlmDatasetPrecached(dataset_file)

    lengths = []

    total_timer = Timer("total")
    timer = Timer("load precached dataset")
    for i, data in enumerate(dataset):
        sentence, labels = data
        if sentence.any():
            nonzero = sentence.nonzero()
            t_max = nonzero.max()
            maximum1 = t_max.item() + 1
            del nonzero
            del t_max
        else:
            maximum1 = 0

        del sentence

        if labels.any():
            nonzero = labels.nonzero()
            t_max = nonzero.max()
            maximum2 = t_max.item() + 1
            del nonzero
            del t_max
        else:
            maximum2 = 0
        del labels

        maximum = max(maximum1, maximum2)
        lengths.append(maximum)

        if (i + 1) % 10000 == 0:
            timer.print(f"Processed {i + 1} samples", restart=True)

    print("done processing tensors")
    lengths = torch.tensor(lengths, dtype=torch.float)
    describe(lengths)
    total_timer.print("total time")


if __name__ == '__main__':
    _main()
