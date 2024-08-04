# BERT / GPT model implementation
## TBD: list sources

# Data and Runtime directories

### movies
The code expects to find a directory named 'movies' under the project dir.
Underneath movies expect


movies
  - vocab
    - vocab.txt  # may be called bert-it-vocab.txt, rename it to vocab.txt
  - datasets
    // xyz is an integer, can have multiple of these sets
    - train_data_xyz.msgpack.gz
    - val_data_xyz.msgpack.gz
    - test_data_xyz.msgpack.gz
  - runs


  During training, the code will create a sub directory under runs, e.g.
  run0, run1, ... each with:

 - run0  
> - logs  
>> - events.timestap.machinename...  # used by tensorboard  
>> - logs.txt  # log file  
> - checkpoints  
>> - checkpoint.py  # checkpoint file  


# Train the model

cd <project-base-dir>
python src/main.py
