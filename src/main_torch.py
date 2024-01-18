import torch
import logging


def ddp_cleanup():
    torch.distributed.destroy_process_group()

def init_mode_set_device(config):
    """
    Initialize parallel mode and device.
    Sets config.run.device
    """
    parallel_mode = config.run.parallel_mode
    if parallel_mode == 'single':
        config.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # config.run.is_primary = True
    elif parallel_mode == 'dp':
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError('DataParallel training requires multiple GPUs.')
        config.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # config.run.is_primary = True
    elif parallel_mode == 'ddp':
        import atexit
        torch.cuda.set_device(config.run.local_rank)
        config.run.device = torch.device('cuda', torch.cuda.current_device())
        atexit.register(ddp_cleanup)
        torch.distributed.init_process_group(backend=config.run.dist_backend)
        # config.run.is_primary = torch.cuda.current_device() == 0
    else:
        raise Exception(f'Unknown parallel mode {parallel_mode}. Valid values are single, dp, ddp.')

def wrap_model(config, model):
    model = model.to(config.run.device)

    mode = config.run.parallel_mode
    if mode == 'single':
        pass
    elif mode == 'dp':
        model = torch.nn.DataParallel(model)
    elif mode == 'ddp':
        logging.debug(f'wrap_model, mode is {mode}, local_rank is {config.run.local_rank}')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.run.local_rank])
    else:
        raise Exception(f'Unknown parallel mode {mode}. Valid values are single, dp, ddp.')
    return model

def get_tokenizer(config):
    if config.run.case == 'movies':
        from transformers import BertTokenizer
        path = config.run.base_dir / 'vocab/bert-it-vocab.txt'
        path = str(path)
        result = BertTokenizer.from_pretrained(path, local_files_only=True)
    elif config.run.case == 'instacart':
        from instacart.instacart_tokenizer import InstacartTokenizer
        path = config.run.base_dir / 'vocab' / 'instacart_vocab.txt'
        result = InstacartTokenizer(path)
    else:
        raise Exception(f'Unknown case. {config.run.case}')
    return result

def get_model(config, tokenizer):
    from bert.bert import BERT
    from bert.bertlm import BERTLM

    model_config = config.model
    vocab_size = len(tokenizer.vocab)

    # todo: if resuming from a ehckpoint, load the model from the checkpoint
    # todo: and reconcile / adjust / validate the model config

    bert_model = BERT(
        vocab_size=vocab_size,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        heads=model_config.heads,
        dropout=model_config.dropout,
        seq_len=model_config.seq_len
    )

    bert_lm = BERTLM(bert_model, vocab_size, apply_softmax=False)

    if config.run.compile:
        # requires PyTorch 2.0
        bert_lm = torch.compile(bert_lm)
        logging.info("Model compiled")
    else:
        logging.info("Model not compiled")

    bert_lm = wrap_model(config, bert_lm)

    return bert_lm

def get_trainer(config, model, tokenizer):
    from bert.trainer import BERTTrainerPreprocessedDatasets
    from bert.trainerB import TrainerB
    modern = True
    if modern:
        trainer = TrainerB(config, model, tokenizer)
    else:
        trainer = BERTTrainerPreprocessedDatasets(
            config,
            model,
            tokenizer=tokenizer
        )
    return trainer

def create_objects(config):
    init_mode_set_device(config)

    tokenizer = get_tokenizer(config)
    model = get_model(config, tokenizer)

    total_parameters = sum([p.nelement() for p in model.parameters()])
    logging.info(f"Total Parameters: {total_parameters:,}")

    trainer = get_trainer(config, model, tokenizer)
    # if self.config.train.checkpoint is not None:
    #     self.trainer.load_checkpoint()
    return trainer
