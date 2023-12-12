import re
import gzip
import torch
import random
import pickle
import msgpack
import itertools
from torch.utils.data import Dataset

from bert.timer import Timer

class BERTDatasetPrecached(Dataset):
    def __init__(self, path):
        self._type = None
        timer = Timer("load precached dataset")
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rb') as f:
            if re.search(r'\.pkl(\.gz)?$', path):
                self._type = 0
                self.cached_data = pickle.load(f)
            elif re.search(r'\.msgpack(\.gz)?$', path):
                self._type = 1
                packed_data = f.read()
                self.cached_data = msgpack.unpackb(packed_data, raw=False)
            else:
                raise ValueError(f"Unknown file type: {path}")
        timer.print(f"loaded precached dataset {path}")

    def __len__(self):
        if self._type == 0:
            return len(self.cached_data)
        elif self._type == 1:
            return len(self.cached_data) // 2
        else:
            raise ValueError(f"Unknown type: {self._type}")
    
    def __getitem__(self, index):
        if self._type == 0:
            return self.cached_data[index]['bert_input'], self.cached_data[index]['bert_label']
        elif self._type == 1:
            l_sentence = self.cached_data[index * 2]
            l_labels = self.cached_data[index * 2 + 1]
            sentence = torch.tensor(l_sentence, dtype=torch.int32)
            labels = torch.tensor(l_labels, dtype=torch.int32)
            del l_sentence
            del l_labels
            return sentence, labels        
        else:
            raise ValueError(f"Unknown type: {self._type}")

class BERTDataset(Dataset):
    def __init__(self, lines, tokenizer, seq_len):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(lines)
        self.lines = lines

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, index):

        # Step 1: get random sentence pair
        line = self.lines[index] 

        # Step 2: replace random words in sentence with mask / random words
        sentence, label = self.random_word2(line)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
         # Adding PAD token for labels
        t1 = [self.tokenizer.vocab['[CLS]']] + sentence + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        bert_input = t1[:self.seq_len]
        bert_label = t1_label[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label


    def random_word2(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # Calculate number of tokens to mask (15% of tokens in the sentence)
        num_to_mask = int(len(tokens) * 0.15)

        # Shuffle indices to randomly select tokens to mask
        indices = list(range(len(tokens)))
        random.shuffle(indices)
        masked_indices = set(indices[:num_to_mask])

        for i, token in enumerate(tokens):
            # Token ID without cls and sep tokens
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if i in masked_indices:
                # Randomly decide the type of masking
                prob = random.random()

                # 80% chance to mask
                if prob < 0.8:
                    output.extend([self.tokenizer.vocab['[MASK]']] * len(token_id))

                # 10% chance to replace with a random token
                elif prob < 0.9:
                    output.extend([random.randrange(len(self.tokenizer.vocab))] * len(token_id))

                # 10% chance to leave the token as it is
                else:
                    output.extend(token_id)

                output_label.extend(token_id)
            else:
                output.extend(token_id)
                output_label.extend([0] * len(token_id))

        # Flattening multi-word tokens
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))

        assert len(output) == len(output_label)
        return output, output_label
