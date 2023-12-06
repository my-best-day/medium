import os
import tqdm
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

# WordPiece tokenizer

### save data as txt file
os.mkdir('./data')

content = []
file_count = 0
def prepare_data(lines):
    for line in tqdm.tqdm(lines):
        content.append(line)

        # once we hit the 10K mark, save to file
        if len(content) == 10000:
            with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(content))
            content = []
            file_count += 1

    paths = [str(x) for x in Path('./data').glob('**/*.txt')]

    ### training own tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train( 
        files=paths,
        vocab_size=30_000, 
        min_frequency=5,
        limit_alphabet=1000, 
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )

    os.mkdir('./bert-it-1')
    tokenizer.save_model('./bert-it-1', 'bert-it')
