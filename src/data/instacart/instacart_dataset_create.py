import random
import msgpack
import pandas as pd
from torch.utils.data import Dataset

from instacart_dataset import InstacartDataset

def read_order_list(path):

    # Read order_token_list from order_item_list.msgpack
    with open('./data/instacart/order_token_list.msgpack', 'rb') as f:
        order_token_list = msgpack.load(f, raw=False)

    return order_token_list

def create_precached_dataset():
    import gzip
    import msgpack
    from instacart_tokenizer import InstacartTokenizer

    from sklearn.model_selection import train_test_split
    random_seed = 271

    orders = read_order_list('./data/instacart/order_token_list.msgpack')
    # orders = orders[:1000]
    train_orders, test_orders = train_test_split(orders, test_size=0.2, random_state=random_seed)
    val_orders, test_orders = train_test_split(test_orders, test_size=0.5, random_state=random_seed)

    tokenizer = InstacartTokenizer('./vocab/instacart_vocab.txt')

    for _ in range(10):
        output = []
        random_seed = random.randint(0, 1000)
        train_dataset = InstacartDataset(train_orders, tokenizer.vocab, 32)
        n = len(train_dataset)
        for i in range(n):
            sentence, label = train_dataset.get_plain(i)
            output.append(sentence)
            output.append(label)
        # print the random seed and the length of the output
        print(f"Random seed: {random_seed}, length: {len(output)}")
        with gzip.open(f'./datasets32/instacart/train_data_{random_seed}.msgpack.gz', 'wb') as f:
            msgpack.dump(output, f)

    output = []
    random_seed = random.randint(0, 1000)
    val_dataset = InstacartDataset(val_orders, tokenizer.vocab, 32)
    n = len(val_dataset)
    for i in range(n):
        sentence, label = val_dataset.get_plain(i)
        output.append(sentence)
        output.append(label)
    # print the random seed and the length of the output
    print(f"Random seed: {random_seed}, length: {len(output)}")
    with gzip.open(f'./datasets32/instacart/val_data_{random_seed}.msgpack.gz', 'wb') as f:
        msgpack.dump(output, f)

    output = []
    random_seed = random.randint(0, 1000)
    test_dataset = InstacartDataset(test_orders, tokenizer.vocab, 32)
    n = len(test_dataset)
    for i in range(n):
        sentence, label = test_dataset.get_plain(i)
        output.append(sentence)
        output.append(label)
    # print the random seed and the length of the output
    print(f"Random seed: {random_seed}, length: {len(output)}")
    with gzip.open(f'./datasets32/instacart/test_data_{random_seed}.msgpack.gz', 'wb') as f:
        msgpack.dump(output, f)

    output = []


def _main():
    create_precached_dataset()

    pass
if __name__ == '__main__':
    _main()