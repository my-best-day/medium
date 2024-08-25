"""
Routines to create objects and trainer, and configure the
training procedure.
"""
import logging
import torch
import inspect
from transformer.task_handler.task_handler_common import TaskHandlerCommon as THC
from transformer.task_handler.mlm_task_handler import MlmTaskHandler


logger = logging.getLogger(__name__)


class TorchConfigurator:
    """
    Create, configure, load, initialize, and synchronize the objects required for training:
    the model, optimizer, and trainer.

    See the `configure` method for the sequence of steps.
    Typical usage:
        configurator = TorchConfigurator(config)
        configurator.configure()
        trainer = configurator.trainer
    """

    def __init__(self, config):
        self.config = config

        self.tokenizer = None
        self.task_handler = None
        self.model = None
        self.optimizer = None
        self.trainer = None

    def configure(self):
        """
        Configure the training process.
        1. figure out what device to use (cpu, cuda)
        2. create the tokenizer
        3. create the task handler which abstract the task-specific details
        4. create the model, optimizer, and trainer
        5. resume from checkpoint if requested
        6. initialize the model and optimizer (if not loaded from checkpoint)
        7. sync objects across processes in DDP mode
        """
        self.init_mode_set_device()

        self.tokenizer = self.create_tokenizer()
        self.task_handler = self.create_task_handler()

        self.create_objects_and_trainer()
        self.resume_from_checkpoint()
        self.initialize_objects()
        self.sync_objects()

    def init_mode_set_device(self):
        """
        Initialize parallel mode and device.
        Sets config.run.device
        """
        parallel_mode = self.config.run.parallel_mode
        if parallel_mode == 'single':
            self.config.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        elif parallel_mode == 'dp':
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError('DataParallel training requires multiple GPUs.')
            self.config.run.device = 'cuda'

        elif parallel_mode == 'ddp':
            if not torch.cuda.is_available():
                raise RuntimeError('DDP requires CUDA devices.')
            torch.cuda.set_device(self.config.run.local_rank)
            self.config.run.device = torch.device('cuda', torch.cuda.current_device())

        else:
            raise ValueError("Unknown parallel mode %s. Valid values are 'single', 'dp', 'ddp'.",
                             parallel_mode)

    def create_tokenizer(self):
        """
        Returns the tokenizer for the given case

        TODO: for GPT use GPT2TokenizerFast with its own vocab file. OK to start with BERT
        tokenizer.
        """
        if self.config.run.case == 'movies' or self.config.run.case == 'dickens':
            from transformers import BertTokenizer
            path = self.config.run.base_dir / 'vocab/'
            path = str(path)
            result = BertTokenizer.from_pretrained(path, local_files_only=True)

        elif self.config.run.case == 'instacart':
            raise NotImplementedError('InstacartTokenizer not implemented')

        else:
            raise ValueError('Unknown case. %s', self.config.run.case)

        return result

    def create_task_handler(self):
        task_type = self.config.model.task_type
        if task_type == 'mlm':
            task_handler = MlmTaskHandler(self.config, self.tokenizer)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        return task_handler

    def create_objects_and_trainer(self):
        # transformer model with a language model head
        self.model = self.create_model()

        self.optimizer = self.create_optimizer()

        self.trainer = self.create_trainer()

    def create_model(self):
        """
        Coordinate the creation of the mode:
        create -> compile -> wrap for ddp
        """
        # model + lm head
        model = self.task_handler.create_lm_model()

        if self.config.run.compile:
            # requires PyTorch 2.0
            model = torch.compile(model)
        logger.info(f"Model {'compiled' if self.config.run.compile else 'not compiled'}")

        model = self.wrap_parallel_model(model)

        total_model_parameters = sum([p.nelement() for p in model.parameters()])
        logger.info(f"Total Model Parameters: {total_model_parameters:,}")

        return model

    def create_transformer_model(self):
        """
        Returns the base, transformer model for the given config and tokenizer.
        """
        from transformer.transformer import Transformer

        model_config = self.config.model
        vocab_size = len(self.tokenizer.vocab)

        transformer_model = Transformer(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            heads=model_config.heads,
            dropout=self.config.train.dropout,
            seq_len=model_config.seq_len,
            is_gpt=self.config.model.task_type == 'gpt',
            use_flash=self.config.run.flash
        )

        return transformer_model

    def wrap_parallel_model(self, model):
        """Wrap the model with Distributed/DataParallel or none according to the config"""
        model = model.to(self.config.run.device)

        mode = self.config.run.parallel_mode
        if mode == 'single':
            # no need to wrap the model
            pass

        elif mode == 'dp':
            model = torch.nn.DataParallel(model)

        elif mode == 'ddp':
            logger.debug('wrap_model, mode is %s, local_rank is %s',
                         mode, self.config.run.local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.config.run.local_rank])

        else:
            raise ValueError('Unknown parallel mode %s (expected: single, dp, ddp).', mode)

        return model

    def create_optimizer(self):
        """
        Returns the optimizer.
        Configure the optimizer with weight decay and no weight decay parameters.
        """

        decay_params, no_decay_params = self.get_decay_no_decay_parameters()

        optimization_groups = [
            {'params': decay_params, 'weight_decay': self.config.train.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        if self.config.run.is_primary:
            self.log_decay_parameter_counts(decay_params, no_decay_params)

        use_fused = self.should_use_fused_adamw()

        if self.config.run.is_primary:
            logger.info("Using fused AdamW: %s", use_fused)

        extra_args = dict()
        if use_fused:
            extra_args['fused'] = True

        optimizer = torch.optim.AdamW(
            optimization_groups,
            lr=self.config.train.learning_rate,
            betas=(0.9, 0.999),
            **extra_args
        )

        return optimizer

    def get_decay_no_decay_parameters(self):
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
        for param in self.model.parameters():
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

    def log_decay_parameter_counts(self, decay_params, no_decay_params):
        """
        Log the number of decayed and non-decayed parameters
        """
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in no_decay_params)
        # pylint: disable=logging-fstring-interpolation
        logger.info(f"num decayed parameter tensors: {len(decay_params)}, "
                    f"with {num_decay_params:,} parameters")
        logger.info(f"num non-decayed parameter tensors: {len(no_decay_params)}, "
                    f"with {num_nodecay_params:,} parameters")

    def should_use_fused_adamw(self):
        """
        Returns True if the fused AdamW optimizer should be used.
        Use fused if configured, available, and device type is 'cuda'
        """
        use_fused = False

        # check if fused is enabled in config
        if self.config.run.fused_adamw:
            # check if fused is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            if fused_available:
                device_type = self.config.run.device \
                    if isinstance(self.config.run.device, str) \
                    else self.config.run.device.type

                # check if device type is 'cuda'
                if device_type == 'cuda':
                    use_fused = True

        return use_fused

    def create_trainer(self):
        from transformer.trainer import Trainer
        trainer = Trainer(self.config, self.model, self.optimizer, self.task_handler)
        return trainer

    def resume_from_checkpoint(self):
        if not self.config.run.is_primary:
            return

        checkpoint_path = self.config.train.checkpoint
        if checkpoint_path is None:
            return

        # we only load the checkpoint in the primary process. A later sync
        # broadcast all parameters / states to other processes.

        logger.info("Resuming from checkpoint at %s", checkpoint_path)
        # use map_location='cpu' if GPU memory an issue (broadcasting required in that case!)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.run.device,
                                weights_only=False)

        self.task_handler.resume_from_checkpoint_dict(self.model, self.optimizer, self.trainer,
                                                      checkpoint)

    def initialize_objects(self):
        """
        Initialize the base model (transformer) and the language model head -
        unless their state was loaded from a checkpoint.
        """
        config = self.config
        if not config.run.is_primary:
            return

        unwrap_model = THC.unwrap_model(self.model)

        if config.run.init_base_model_weights:
            base_model = unwrap_model.base_model
            self.task_handler.init_base_model_weights(base_model)

        if config.run.init_lm_head_weights:
            lm_head = unwrap_model.lm_head
            self.task_handler.init_lm_head_weights(lm_head)

    def sync_objects(self):
        """
        Sync objects across processes in DDP mode.

        sync the model parameters (base + lm head) either if was loaded from a checkpoint
        or if the model was initialized from scratch.

        sync the optimizer state if it was loaded from a checkpoint. Otherwise, the optimizer
        initialization is deterministic and does not require synchronization.

        sync of the trainer state is quite trivial, so we always do it.
        """
        config = self.config

        if config.run.parallel_mode != 'ddp':
            return

        for param in self.model.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # sync optimizer state
        if not config.run.skip_sync_optimizer_state:
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        torch.distributed.broadcast(param.data, src=0)
            torch.distributed.broadcast(self.optimizer.state, src=0)

        # ensure synchronization across all ranks
        torch.distributed.barrier()

        # sync trainer state
        self.sync_trainer_state()

    def sync_optimizer(self):
        """
        Sync the optimizer state across all ranks
        """

        # The optimizer params are the model params. We sync the model params
        # earlier in the process, so there is no need to resync them here.

        for _, state in self.optimizer.state.items():
            if isinstance(state, torch.Tensor):
                torch.distributed.broadcast(state, src=0)
            elif isinstance(state, dict):
                # handle nested state
                for _, sub_state in state.items():
                    if isinstance(sub_state, torch.Tensor):
                        torch.distributed.broadcast(sub_state, src=0)

    def sync_trainer_state(self):
        """ Sync the trainer state across all ranks """
        device = self.config.run.device
        pt_sample_iter_start = \
            torch.tensor(self.trainer.sample_iter_start, dtype=torch.long).to(device)
        pt_sample_iter = \
            torch.tensor(self.trainer.sample_iter, dtype=torch.long).to(device)
        pt_start_iter = \
            torch.tensor(self.trainer.start_iter, dtype=torch.long).to(device)
        pt_iter = \
            torch.tensor(self.trainer.iter, dtype=torch.long).to(device)
        pt_best_val_loss = \
            torch.tensor(self.trainer.best_val_loss, dtype=torch.float).to(device)

        torch.distributed.broadcast(pt_sample_iter_start, src=0)
        torch.distributed.broadcast(pt_sample_iter, src=0)
        torch.distributed.broadcast(pt_start_iter, src=0)
        torch.distributed.broadcast(pt_iter, src=0)
        torch.distributed.broadcast(pt_best_val_loss, src=0)

        # ensure synchronization across all ranks
        torch.distributed.barrier()

        self.trainer.sample_iter_start = pt_sample_iter_start.item()
        self.trainer.sample_iter = pt_sample_iter.item()
        self.trainer.start_iter = pt_start_iter.item()
        self.trainer.iter = pt_iter.item()
        self.trainer.best_val_loss = pt_best_val_loss.item()
