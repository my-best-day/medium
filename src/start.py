import os
import torch
import logging
from pathlib import Path

from args import get_args
from utils.logging import config_logging
from args_to_config import get_config

class Start:
    def __init__(self, config):
        self.config = config
        # self.init_mode_set_device()

    def train(self):
        logging.debug("loading tokenizer")
        self.tokenizer = self.get_tokenizer()
        logging.debug("got tokenizer")

        logging.debug("getting model")
        model = self.get_model(self.tokenizer)
        logging.debug("wrapping model")
        self.model = self.wrap_model(model)
        logging.debug("wrapped model")

        total_parameters = sum([p.nelement() for p in self.model.parameters()])
        logging.info(f"Total Parameters: {total_parameters:,}")

        self.trainer = self.get_trainer()
        if self.config.train.checkpoint is not None:
            self.trainer.load_checkpoint()

        self.trainer.train()

    def get_tokenizer(self):
        if self.config.run.case == 'movies':
            from transformers import BertTokenizer
            path = self.config.run.base_dir / 'vocab/bert-it-vocab.txt'
            path = str(path)
            result = BertTokenizer.from_pretrained(path, local_files_only=True)
        elif self.config.run.case == 'instacart':
            from instacart.instacart_tokenizer import InstacartTokenizer
            path = self.config.run.base_dir / 'vocab' / 'instacart_vocab.txt'
            result = InstacartTokenizer(path)
        else:
            raise Exception(f'Unknown case. {self.config.run.case}')
        return result

    def get_model(self, tokenizer):
        from bert.bert import BERT
        from bert.bertlm import BERTLM

        model_config = self.config.model
        vocab_size = len(tokenizer.vocab)

        bert_model = BERT(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            heads=model_config.heads,
            dropout=model_config.dropout,
            seq_len=model_config.seq_len
        )

        bert_lm = BERTLM(bert_model, vocab_size, apply_softmax=True)

        return bert_lm

    def wrap_model(self, model):
        mode = self.config.run.parallel_mode
        model = model.to(self.config.run.device)
        if mode == 'single':
            pass
        elif mode == 'dp':
            model = torch.nn.DataParallel(model)
        elif mode == 'ddp':
            logging.debug(f'wrap_model, mode is {mode}, local_rank is {self.config.run.local_rank}')
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.config.run.local_rank])
            logging.debug("wrap_model: model wrapped")
        else:
            raise Exception(f'Unknown parallel mode {mode}. Valid values are single, dp, ddp.')
        return model

    def get_trainer(self):
        from bert.trainer import BERTTrainerPreprocessedDatasets

        trainer = BERTTrainerPreprocessedDatasets(
            self.config,
            self.model,
            tokenizer=self.tokenizer
        )
        return trainer


def init_mode_set_device(config):
    """
    Initialize parallel mode and device.
    Sets config.run.device
    """
    parallel_mode = config.run.parallel_mode
    if parallel_mode == 'single':
        config.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif parallel_mode == 'dp':
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError('DataParallel training requires multiple GPUs.')
        config.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif parallel_mode == 'ddp':
        os.environ['MASTER_ADDR'] = config.run.dist_master_addr
        os.environ['MASTER_PORT'] = config.run.dist_master_port
        torch.cuda.set_device(config.run.local_rank)
        config.run.device = torch.device('cuda', torch.cuda.current_device())
    else:
        raise Exception(f'Unknown parallel mode {parallel_mode}. Valid values are single, dp, ddp.')
    # this is the primary node if we are using the first gpu or if we are not using gpus
    config.run.is_primary = not torch.cuda.is_available() or torch.cuda.current_device() == 0

def _main():
    args = get_args()
    config = get_config(args)

    init_mode_set_device(config)

    logfile_path = config.run.logs_dir / 'log.txt'
    config_logging(logfile_path)

    if config.run.is_primary:
        logging.info(config.model)
        logging.info(config.train)
        logging.info(config.run)

        if config.run.wandb:
            config_wandb(config)

    if config.run.parallel_mode == 'ddp':
        torch.distributed.init_process_group(backend=config.run.dist_backend, init_method='env://')

    try:
        logging.debug("creating start")
        start = Start(config)
        logging.debug("created start")
        start.train()
        logging.debug("done training")
    except Exception as e:
        logging.exception(e)
        raise e

def config_wandb(config):
    import wandb
    wandb.init(
        project=config.run.case,
        config=config.to_dict(),
        name=f'run{config.run.run_id}',
        dir=config.run.run_dir,
    )


if __name__ == '__main__':
    _main()