"""
Routines to create objects and trainer, and configure the
training procedure.
"""
import logging
import torch
import inspect


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


def wrap_parallel_model(config, model):
    """Wrap the model with Distributed/DataParalel or none according to the config"""
    model = model.to(config.run.device)

    mode = config.run.parallel_mode
    if mode == 'single':
        # do nothing
        pass
    elif mode == 'dp':
        model = torch.nn.DataParallel(model)
    elif mode == 'ddp':
        logging.debug('wrap_model, mode is %s, local_rank is %s', mode, config.run.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.run.local_rank])
    else:
        raise ValueError('Unknown parallel mode %s (expected: single, dp, ddp).', mode)
    return model


def get_tokenizer(config):
    """Returns the tokenizer for the given case"""
    if config.run.case == 'movies' or config.run.case == 'dickens':
        from transformers import BertTokenizer
        path = config.run.base_dir / 'vocab/'
        path = str(path)
        result = BertTokenizer.from_pretrained(path, local_files_only=True)
    elif config.run.case == 'instacart':
        raise NotImplementedError('InstacartTokenizer not implemented')
        # NOSONAR
        # from instacart.instacart_tokenizer import InstacartTokenizer
        # path = config.run.base_dir / 'vocab' / 'instacart_vocab.txt'
        # result = InstacartTokenizer(path)
    else:
        raise ValueError('Unknown case. %s', config.run.case)
    return result


def get_model(config, tokenizer):
    """
    Returns the language model for the given config and tokenizer.
    TODO: handle GPT
    """
    model = get_bert_model(config, tokenizer)

    if config.run.compile:
        # requires PyTorch 2.0
        model = torch.compile(model)
        logging.info("Model compiled")
    else:
        logging.info("Model not compiled")

    model = wrap_parallel_model(config, model)

    return model


def get_bert_model(config, tokenizer):
    from bert.bert import BERT
    from bert.bertlm import BERTLM
    from bert.lm.classifier.bert_classifier_model import BertClassifierModel

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

    if config.model.task_type == 'cola':
        bert_lm = BertClassifierModel(bert_model, 2)
    elif config.model.task_type == 'mlm':
        bert_lm = BERTLM(bert_model, vocab_size, apply_softmax=False)
    else:
        raise ValueError('Unknown task type %s', config.model.task_type)

    return bert_lm


def get_trainer(config, model, optimizer, tokenizer):
    from bert.trainer import Trainer
    trainer = Trainer(config, model, optimizer, tokenizer)
    return trainer


def create_objects_and_trainer(config):
    init_mode_set_device(config)

    # right tokenizer by case (movies, instacart, dickens)
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

    if config.model.task_type != 'mlm':
        # Filter out 'mask_lm' parameters
        model_state = {k: v for k, v in checkpoint['model'].items() if 'mask_lm' not in k}

    is_wrapped = is_model_wrapped(config)
    # in DDP, load state dict into the underlying model, otherwise, load it directly
    (model.module if is_wrapped else model).load_state_dict(model_state, strict=False)

    if config.run.parallel_mode == 'ddp':
        for param in model.module.parameters():
            torch.distributed.broadcast(param.data, src=0)

    if config.model.task_type == 'mlm':
        # load optimizer state
        optimizer_state = checkpoint['optimizer']
        optimizer.load_state_dict(optimizer_state)

        # load trainer state
        iteration = checkpoint['iter']
        val_loss = checkpoint['val_loss']
    else:
        iteration = 0
        val_loss = 0

    trainer.start_iter = iteration
    trainer.iter = trainer.start_iter

    trainer.best_val_loss = val_loss

    logging.info("Resuming from iteration %s, with val loss %s", iteration, val_loss)


def is_model_wrapped(config):
    result = config.run.parallel_mode in ('dp', 'ddp')
    return result


def configure_optimizer(config, model):

    decay_params, no_decay_params = get_decay_no_decay_parameters(model)

    optimization_groups = [
        {'params': decay_params, 'weight_decay': config.train.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    if config.run.is_primary:
        log_decay_parameter_counts(decay_params, no_decay_params)

    use_fused = should_use_fused_adamw(config)

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


def get_decay_no_decay_parameters(model):
    """
    Figure out which parameters require weight decay and which do not.
    Iterates over the parameters of the model, separates them into two lists.

    Parameters:
        model (torch.nn.Module): The model whose parameters need to be analyzed.

    Returns:
        tuple: A tuple containing two lists:
            - decay_params (list): Parameters that require weight decay.
            - no_decay_params (list): Parameters that do not require weight decay.

    Parameters with a shape of (1,) or (1, n) do not need weight decay.
    These are typically bias terms or scalar parameters, which do not benefit
    from regularization through weight decay.
    """
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
    return decay_params, no_decay_params


def should_use_fused_adamw(config):
    """
    Returns True if the fused AdamW optimizer should be used.
    Use fused if configured, available, and device type is 'cuda'
    """
    use_fused = False

    # check if fused is enabled in config
    if config.run.fused_adamw:
        # check if fused is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        if fused_available:
            device_type = config.run.device if isinstance(config.run.device, str) else \
                config.run.device.type
            # check if device type is 'cuda'
            if device_type == 'cuda':
                use_fused = True

    return use_fused


def log_decay_parameter_counts(decay_params, no_decay_params):
    """
    Log the number of decayed and non-decayed parameters
    """
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in no_decay_params)
    # pylint: disable=logging-fstring-interpolation
    logging.info(f"num decayed parameter tensors: {len(decay_params)}, "
                 f"with {num_decay_params:,} parameters")
    logging.info(f"num non-decayed parameter tensors: {len(no_decay_params)}, "
                 f"with {num_nodecay_params:,} parameters")
