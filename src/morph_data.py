import gzip
import torch
import random
import pickle
import msgpack
import itertools
from bert.timer import Timer


def morph_data(in_path):
    # out_path = in_path - {'.pkl', '.pkl.gz'} + '.msgpack'
    for ext in ['.pkl', '.pkl.gz']:
        if in_path.endswith(ext):
            out_path = in_path[:-len(ext)]
            break
    out_path = out_path + '.msgpack'
    
    timer = Timer("load precached dataset")
    opener = gzip.open if in_path.endswith('.gz') else open
    with opener(in_path, 'rb') as f:
        data = pickle.load(f)
    timer.print("loaded precached dataset", True)

    lists =[]
    for record in data:
        lists.append(record['bert_input'].tolist())
        lists.append(record['bert_label'].tolist())
    timer.print("converted to lists", True)
    
    with open(out_path, 'wb') as file:
        packed_data = msgpack.packb(lists)
        file.write(packed_data)
    timer.print("wrote msgpack", True)

    with gzip.open(f"{out_path}.gz", "wb") as file:
        packed_data = msgpack.packb(lists)
        file.write(packed_data)
    timer.print("wrote msgpack.gz", True)

morph_data('./datasets/train_data_177.pkl.gz')
morph_data('./datasets/train_data_12.pkl')


