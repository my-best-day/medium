"""
TBD
"""
import logging
import torch


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
        config.run.device = 'cuda'
    elif parallel_mode == 'ddp':
        torch.cuda.set_device(config.run.local_rank)
        config.run.device = torch.device('cuda', torch.cuda.current_device())
    else:
        # pylint: disable=exception-arguments
        raise ValueError("Unknown parallel mode %s. Valid values are 'single', 'dp', 'ddp'.",
                         parallel_mode)


def wrap_model(config, model):
    """Wrap the model with Distributed/DataParalel or none according to the config"""
    model = model.to(config.run.device)

    mode = config.run.parallel_mode
    if mode == 'single':
        pass
    elif mode == 'dp':
        model = torch.nn.DataParallel(model)
    elif mode == 'ddp':
        logging.debug('wrap_model, mode is %s, local_rank is %s', mode, config.run.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.run.local_rank])
    else:
        raise Exception('Unknown parallel mode %s. Valid values are single, dp, ddp.', mode)
    return model


def get_tokenizer(config):
    """Returns the tokenizer for the given case"""
    if config.run.case == 'movies':
        from transformers import BertTokenizer
        path = config.run.base_dir / 'vocab/'  # bert-it-vocab.txt'
        path = str(path)
        result = BertTokenizer.from_pretrained(path, local_files_only=True)
    elif config.run.case == 'instacart':
        from instacart.instacart_tokenizer import InstacartTokenizer
        path = config.run.base_dir / 'vocab' / 'instacart_vocab.txt'
        result = InstacartTokenizer(path)
    elif config.run.case == 'charles':
        from transformers import GPT2Tokenizer
        result = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        raise ValueError('Unknown case. %s', config.run.case)
    return result


def get_model(config, tokenizer):
    """
    Returns the language model for the given config and tokenizer.
    TODO: handle GPT
    """
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
        dropout=config.train.dropout,
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


def get_trainer(config, model, optimizer, tokenizer):
    from bert.trainer import BERTTrainerPreprocessedDatasets
    from bert.trainerB import TrainerB
    modern = True
    if modern:
        trainer = TrainerB(config, model, optimizer, tokenizer)
    else:
        trainer = BERTTrainerPreprocessedDatasets(
            config,
            model,
            optimizer,
            tokenizer=tokenizer
        )
    return trainer


def create_objects_and_trainer(config):
    init_mode_set_device(config)

    # right tokenizer by case (movies, instacart, charles)
    tokenizer = get_tokenizer(config)

    # BERT lang model / GPT (pending...)
    model = get_model(config, tokenizer)

    total_model_parameters = sum([p.nelement() for p in model.parameters()])
    # pylint: disable=logging-fstring-interpolation
    logging.info(f"Total Model Parameters: {total_model_parameters:,}")

    optimizer = configure_optimizer(config, model)
    trainer = get_trainer(config, model, optimizer, tokenizer)

    resume_from_checkpoint(config, model, optimizer, trainer)

    return trainer


def resume_from_checkpoint(config, model, optimizer, trainer):
    path = config.train.checkpoint
    if path is None:
        return

    logging.info("Resuming from checkpoint at %s", path)
    # use map_location='cpu' if GPU memory an issue (broadcasting required in that case!)
    checkpoint = torch.load(path, map_location=config.run.device)

    # load model state
    model_state = checkpoint['model']
    is_wrapped = is_model_wrapped(config)
    # in DDP, load state dict into the underlying model, otherwise, load it directly
    (model.module if is_wrapped else model).load_state_dict(model_state)

    if config.run.parallel_mode == 'ddp':
        for param in model.module.parameters():
            torch.distributed.broadcast(param.data, src=0)

    # load optimizer state
    optimizer_state = checkpoint['optimizer']
    optimizer.load_state_dict(optimizer_state)

    # load trainer state
    iteration = checkpoint['iter']
    trainer.start_iter = iteration
    trainer.iter = trainer.start_iter

    val_loss = checkpoint['val_loss']
    trainer.best_val_loss = val_loss

    logging.info("Resuming from iteration %s, with val loss %s", iteration, val_loss)


def is_model_wrapped(config):
    result = config.run.parallel_mode in ('dp', 'ddp')
    return result


def configure_optimizer(config, model):
    import inspect

    # figure which parameters require weight decay
    seen = set()
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param in seen:
            continue
        seen.add(param)
        if len(param.shape) == 1 or param.shape[0] == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimization_groups = [
        {'params': decay_params, 'weight_decay': config.train.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    if config.run.is_primary:
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in no_decay_params)
        # pylint: disable=logging-fstring-interpolation
        logging.info(f"num decayed parameter tensors: {len(decay_params)}, "
                     f"with {num_decay_params:,} parameters")
        logging.info(f"num non-decayed parameter tensors: {len(no_decay_params)}, "
                     f"with {num_nodecay_params:,} parameters")

    # use fused if available and or device type is 'cuda'
    if config.run.fused_adamw:
        fused_available_ = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        device_type = config.run.device if isinstance(config.run.device, str) else \
            config.run.device.type
        use_fused = fused_available_ and device_type == 'cuda'
    else:
        use_fused = False

    if config.run.is_primary:
        logging.info("Using fused AdamW: %s", use_fused)

    extra_args = dict()
    if use_fused:
        extra_args['fused'] = True

    optimizer = torch.optim.AdamW(
        optimization_groups,
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        **extra_args
    )

    return optimizer
