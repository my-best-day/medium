# BERT Model and Training

#### Based on and inspired by [Andrej Kapathy's nonoGPT](https://github.com/karpathy/nanoGPT)

I built this project in order to learn by doing the anatomy and implementation of the transformer architecture. Right now, the focus here in on the BERT model. 

#### Let's start with some results:

![Validation accuracy approaches 60%!](./etc/assets/val_accuracy_20240805.png)

We achieved MLM accuracy approaching **60%** after pre-training from scratch on the WikiText-103 dataset. This is considered good results. 

Here is a cherry picked example. The first block shows the predictions inside /slashes/. The second block is the original text.

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

In this example the model guessed 13 out of the 15 masked tokens, an accuracy of 80%.

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
We don't see here a few of the other parameters, for example we used 2 micro-steps.

## To train the model

### Dataset Preprocessing
Use the script at [scripts/prepare_mlm_dataset.py](./scripts/prepare_mlm_dataset.py) to generate a preprocessed MLM dataset in the format that the trainer expects. In this dataset we mask about 15% of the tokens in each "sentence" (chunk of sentences).  

The input file is a single text file of english words. For example, download (and extract)
the WikiText-103 dataset.

Prepare the dataset using scripts/prepare_mlm_dataset.py

```
python scripts/prepare_mlm_dataset.py -h
usage: prepare_mlm_dataset.py [-h] -i INPUT -l LABEL -o OUTPUT -m MAX_LEN -v VOCAB [-s SEED]

Prepare MLM dataset for pre-training of a BERT model.

Example usage:
python prepare_mlm_dataset.py -i wiki.train.tokens -l wiki -o wiki/datasets
                                -m 128 -v wiki/vocab

This will generate the following files:
    wiki/datasets/train_wiki_128_123.msgpack.gz
    wiki/datasets/val_wiki_128_123.msgpack.gz
    wiki/datasets/test_wiki_128_123.msgpack.gz
where wiki is the label, 128 is the max length, and 123 is the random seed.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The input text file to be processed.
  -l LABEL, --label LABEL
                        The label to be used for the dataset.
  -o OUTPUT, --output OUTPUT
                        The output directory for the dataset.
  -m MAX_LEN, --max-len MAX_LEN
                        The maximum length of the tokens.
  -v VOCAB, --vocab VOCAB
                        The directory containing the vocabulary files.
  -s SEED, --seed SEED  The random seed to be used.
```

### The training process expects the following directory structure
`<project-base>`  
scripts...   
src...  
wiki
  - vocab
    - vocab.txt  &nbsp;&nbsp;&nbsp;// source to be determined...
  - datasets   
    // the files generated by prepare_mlm_dataset.py described above 
    - train_data_xyz.msgpack.gz
    - val_data_xyz.msgpack.gz
    - test_data_xyz.msgpack.gz
  - runs
    - run0 &nbsp;&nbsp;&nbsp;// these directories are created during training
      - logs
        - log.txt
        - events log file
      - checkpoints
         checkpint.pt


### Todo:
* Adjust the model to also support the GPT model and train it
* Fine tune the model. So far I only pre-trained it
* Create a inference use case example
* Consider adopting more changes from DistilBERT

### movies
The code expects to find a directory named 'movies' under the project dir.
Underneath movies expect
