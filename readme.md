# BERT Model and Training

#### Based on and inspired by [Andrej Karpathy's nonoGPT](https://github.com/karpathy/nanoGPT)

I built this project in order to learn by doing the anatomy and implementation of the transformer architecture. Right now, the focus is on the BERT model. 

#### Let's start with some results:

![Validation accuracy approaches 60%!](./etc/assets/MLM_val_accuracy.png)

We achieved MLM accuracy **exceeding 60%** after pre-training from scratch on the WikiText-103 dataset. This is considered a good result. 

Here is a cherry-picked example. The first block shows the predictions inside /slashes/. The second block is the original text.

---

he fire fighters 
<span style="background-color: lightgreen;">**/./**</span> 
after seeing how upset susan is about the school , ben confesses 
<span style="background-color: lightgreen;">**/to/**</span> 
<span style="background-color: orange;">~~/to/~~</span> the fire . 
<span style="background-color: orange;">~~/she/~~</span> later learns that the school has 
<span style="background-color: lightgreen;">**/somehow/**</span> saved . reception 
<span style="background-color: lightgreen;">**/acc/**</span> 
<span style="background-color: lightgreen;">**/##ola/**</span> des woodburne has earned various award nominations for her role as susan . in 200 
<span style="background-color: orange;">~~/##7/~~</span> , she 
<span style="background-color: lightgreen;">**/was/**</span> /nominated/ for best female performance in a soap from 
<span style="background-color: lightgreen;">**/the/**</span> rose d 
<span style="background-color: lightgreen;">**/'/**</span> or awards . at the 2007 inside soap awards , wood 
<span style="background-color: lightgreen;">**/##burn/**</span>e was nominated 
<span style="background-color: lightgreen;">**/for/**</span> best actress , best couple with alan fletcher and best storyline for susan 
<span style="background-color: lightgreen;">**/and/**</span> karl 
<span style="background-color: lightgreen;">**/'/**</span> s wedding . t

---

he fire fighters 
<span style="background-color: lightgreen;">**.**</span> after seeing how upset susan is about the school , ben confesses 
<span style="background-color: lightgreen;">**to**</span> 
<span style="background-color: orange;">~~starting~~</span> the fire . 
<span style="background-color: orange;">~~susan~~</span> later learns that the school has <span style="background-color: lightgreen;">somehow</span> saved . reception 
<span style="background-color: lightgreen;">**acc**</span>
<span style="background-color: lightgreen;">**ola**</span> des woodburne has earned various award nominations for her role as susan . in 200
<<span style="background-color: orange;">~~5~~</span> , she 
<span style="background-color: lightgreen;">**was**</span> nominated for best female performance in a soap from 
<span style="background-color: lightgreen;">**the**</span> rose d 
<span style="background-color: lightgreen;">**'**</span> or awards . at the 2007 inside soap awards , wood
<span style="background-color: lightgreen;">**burne**</span> was nominated 
<span style="background-color: lightgreen;">**for**</span> best actress , best couple with alan fletcher and best storyline for susan 
<span style="background-color: lightgreen;">**and**</span> karl 
<span style="background-color: lightgreen;">**'**</span> s wedding . t

---

In this carefully selected example the model guessed 13 out of the 15 masked tokens, an accuracy of 87%. 

#### Hyperparameters 
The above results were generated using the following configuration:   
```
seq-len = 128  
batch-size = 200  
val-interval = 100
d-model = 768
heads = 12
n-layer = 6
dropout = 0.35

async-to-device: true
fused-adamw: true
compile: true

learning-rate = 1.0e-3
min-learning-rate = 5e-5
warmup-iters = 1000
lr-decay-iters = 45_000
max-iters = 60_000
weight-decay = 0.01
```

Adjust the dataset patterns towards the bottom of the config file to match your datasets.

Note that a few parameters are not listed here  
For now, you can adjust the number of micro-steps within src/bert/trainer.py, acount line 50:  
`self.micro_step_count = 2`

## Training the Model

### Dataset Generation

See [here](./dataset_preperation.md) a discussion of two methods of generating precached datasets. 

#### Download Dataset 
I used WikiText-103 which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/dekomposition/wikitext103).

This dataset is quite big, if you just toying arround, I suggest you chop it up:  
`tail -n 180k ignore/wiki/wiki.train.tokens > ignore/wiki/wiki.train.tokens.180`

#### Generation Scripts

There are two sets of utilities to generate MLM precached datasets. 

#### Text Segmentation
Text segmentation is [explined here](./dataset_preperation.md#text-segmentation).

Use:  [prepare_mlm_dataset.py](./scripts/prepare_mlm_dataset.py)  

`prepare_mlm_dataset.py [-h] -i INPUT -l LABEL -o OUTPUT -m MAX_LEN -v VOCAB [-s SEED]`

Example:  
`python prepare_mlm_dataset.py -i wiki.train.tokens -l wiki -o wiki/datasets
                                -m 128 -v wiki/vocab`

This will generate three files:
* `wiki/datasets/train_wiki_128_123.msgpack.gz`
* `wiki/datasets/val_wiki_128_123.msgpack.gz`
* `wiki/datasets/test_wiki_128_123.msgpack.gz`

128 is sequence length  
123 is the seed of the random number generator  
wiki is the label

#### Token Splitting

#### TBD

See:  
[tokenize_text.py](./scripts/tokenize_text.py), and  
[prepare_mlm_fixed_len_dataset_from_ids.py](./scripts/prepare_mlm_fixed_len_dataset_from_ids.py).

tokenize_text.py convert the text to a list of token ids.  
prepare_mlm_fixed_len_dataset_from_ids.py uses the output of tokenize_text to create samples by token splitting.

### How to run the Training
pip install -r requirements.txt

```sh
mkdir wiki wiki/input wiki/vocab wiki/datasets wiki/runs
```

#### Download a vocab.txt
For example from [Hugging Face](https://huggingface.co/google-bert/bert-base-uncased/tree/main). I used both this vocab and another one, smaller. Training runs faster, consuming less memory, with a smaller vocab, and in my case, the smaller vocab yielded better results. 


`cp etc/templates/config_template.ini config.ini`

if you are running on a machine without gpu/cuda, also copy:  
`cp etc/tempaltes/local_config_template.ini local_config.ini`

if you're using `local_config.ini`, edit that file, otherwise edit `config.ini`

`seq-len` needs to match the max len of your dataset  
`batch-size` needs to fit the memory of your GPU  
`case = dickens` for now, will be changed to mlm later  
`base-dir = wiki`  

start the training:  
```sh
python src/main.py
```

if you're lucky and have multiple gpus you may use:  
```sh
python src/main.py --ddp --nproc 2
```  
adjust nproc to your gpu count

### Your directory structure should look something like:
```
<project-base>  
scripts...   
src...  
wiki
  - vocab
    - vocab.txt  // source to be determined...
  - datasets   
    // the files generated by prepare_mlm_dataset.py described above 
    - train_data_xyz.msgpack.gz
    - val_data_xyz.msgpack.gz
    - test_data_xyz.msgpack.gz
  - runs
    - run0    // these directories are created during training
      - logs
        - log.txt
        - events log file
      - checkpoints
         checkpoint.pt
  ```



### Todo:
* Adjust the model to also support the GPT model and train it
* Fine tune the model. So far I only pre-trained it
* Create a inference use case example
* Consider adopting more changes from DistilBERT
