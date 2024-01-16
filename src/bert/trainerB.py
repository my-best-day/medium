import time
import torch
import logging

from contextlib import nullcontext

from mtimer import MTimer
from bert.timer import Timer


# TODO: adjust eval_interval, max_iters, val_iters by world_size (number of GPUs)

class TrainerB:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        self.optimizer = self.configure_optimizer(model)

        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        # 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        device_type = self.config.run.device if type(self.config.run.device) == str else self.config.run.device.type
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.autocast_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.train_loader_iter = None
        self.val_loader_iter = None

        self.dataset_counters = {'train': 0, 'val': 0}

        # todo: take from config
        self.iter = 0
        self.max_iters = 5000
        self.micro_step_count = 1
        self.grad_clip = 1.0
        self.val_iters = 10

    def train(self):
        timer = Timer()
        losses = []
        X, Y = self.get_batch('train')
        while self.should_continue_training():
            if self.should_estimate_loss():
                elapsed = timer.elapsed(restart=False)
                self.estimate_loss_and_log_progress(elapsed, losses)
                losses = []
            # enter micro-step
            accumulated_loss = 0.0
            for micro_step in range(self.micro_step_count):
                logits, loss = self.forward_and_loss(micro_step, X, Y)
                X, Y = self.get_batch('train')
                loss /= self.micro_step_count
                accumulated_loss += loss
                self.backward(loss)
            # outside micro-step
            logging.debug(f"loss: {accumulated_loss}")
            losses.append(accumulated_loss)
            self.step()
            self.optimizer.zero_grad()
            self.iter += 1

    def should_continue_training(self):
        return self.iter < self.max_iters

    def should_estimate_loss(self):
        return self.iter % self.config.train.val_interval == 0

    def forward_and_loss(self, micro_step, X, Y):
        no_sync_ctx = self.model.no_sync() if (micro_step != self.micro_step_count - 1) else nullcontext()
        with self.autocast_ctx, no_sync_ctx:
            logits = self.model(X)
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), Y, ignore_index=0)
        return logits, loss

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self):
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()


    # def get_batch(self, split):
    #     if split == 'train':
    #         loader_iter = self.train_loader_iter
    #     elif split == 'val':
    #         loader_iter = self.val_loader_iter
    #     else:
    #         raise ValueError(f"Unknown split: {split}")
    #     X, Y = next(loader_iter)
    #     X = X.to(self.config.run.device)
    #     Y = Y.to(self.config.run.device)
    #     return X, Y

    def estimate_loss_and_log_progress(self, elapsed, train_losses):
        train_loss = sum(train_losses) / len(train_losses) if len(train_losses) > 0 else None
        val_loss, val_accuracy = self.estimate_val_loss()
        if self.config.run.is_primary:
            self.log_progress(elapsed, train_loss, val_loss, val_accuracy)

    def log_progress(self, elapsed, train_loss, val_loss, val_accuracy):
        self.log_tensorboard(train_loss, val_loss, val_accuracy)

        if self.config.run.wandb:
            self.log_wandb(train_loss, val_loss, val_accuracy)

        num_digits = len(f'{self.max_iters:,}')
        per_iteration = elapsed / self.iter if self.iter > 0 else 100.0
        remaining = per_iteration * (self.max_iters - self.iter)
        loss_str = 'None' if train_loss is None else f'{train_loss:5.2f}'
        items = [
            f'{self.iter / self.max_iters:4.0%}',
            f'{self.iter:>{num_digits},}/{self.max_iters:,}',
            f'e: {time.strftime("%H:%M:%S", time.gmtime(elapsed))}',
            f'r: {time.strftime("%H:%M:%S", time.gmtime(remaining))}',
            f'{time.strftime("%M:%S", time.gmtime(per_iteration * 1000.0))} /Kit',
            f't.loss: {loss_str}',
            f'v.loss: {val_loss:5.2f}',
            f'v.accu: {val_accuracy:.1%}',
        ]
        msg = ' | '.join(items)
        logging.info(msg)

    def log_tensorboard(self, train_loss, val_loss, val_accuracy):
        if train_loss is not None:
            self.writer.add_scalar('train_loss', train_loss, self.iter)
        self.writer.add_scalar('val_loss', val_loss, self.iter)
        self.writer.add_scalar('val_accuracy', val_accuracy, self.iter)

    def log_wandb(self, train_loss, val_loss, val_accuracy):
        import wandb
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy}, step=self.iter)

    @torch.no_grad()
    def estimate_val_loss(self):
        losses = []
        total = 0
        correct = 0
        self.model.eval()

        for val_iter in range(self.val_iters):
            X, Y = self.get_batch('val')
            logits = self.model(X)
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), Y, ignore_index=0)
            losses.append(loss)
            logging.debug('val loss: %f', loss)

            probabilities = torch.softmax(logits, dim=-1)
            _, predicted = torch.max(probabilities, dim=-1)
            total += Y.size(0) * Y.size(1)
            correct += torch.sum(Y == predicted).item()

        self.model.train()
        val_loss = sum(losses) / len(losses)
        val_accuracy = correct / total

        val_loss = self.sync_up(val_loss)
        val_accuracy = self.sync_up(val_accuracy)

        return val_loss, val_accuracy

    def sync_up(self, item):
        if self.config.run.parallel_mode == 'ddp':
            if isinstance(item, torch.Tensor):
                tensor = item
            else:
                tensor = torch.tensor(item).to(self.config.run.device)
            torch.distributed.all_reduce(tensor)
            result = tensor.item() / torch.distributed.get_world_size()
        else:
            result = item.item() if isinstance(item, torch.Tensor) else item
        return result

    @property
    def writer(self):
        from torch.utils.tensorboard import SummaryWriter
        if not hasattr(self, '_writer'):
            self._writer = SummaryWriter(str(self.config.run.logs_dir))
        return self._writer

    def configure_optimizer(self, model):
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
            {'params': decay_params, 'weight_decay': self.config.train.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        if self.config.run.is_primary:
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in no_decay_params)
            logging.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            logging.info(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_nodecay_params:,} parameters")

        fused_available_ = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use fused if available and or device type is 'cuda'
        device_type = self.config.run.device if type(self.config.run.device) == str else self.config.run.device.type
        use_fused = fused_available_ and device_type == 'cuda'

        if self.config.run.is_primary:
            logging.info(f"Using fused AdamW: {use_fused}")

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

    # TODO: better scheme for iteration that works with DDP
    def get_batch(self, split):
        if split == 'train':
            iter = self.train_loader_iter
        elif split == 'val':
            iter = self.val_loader_iter
        else:
            raise ValueError(f"Unknown split: {split}")

        if iter is None:
            iter = self.get_data_iter(split)

        try:
            X, Y = next(iter)
            X = X.to(self.config.run.device)
            Y = Y.to(self.config.run.device)
            return X, Y
        except StopIteration:
            # Handle the case when the iterator is exhausted
            # For example, you can reset the iterator or raise an exception
            if split == 'train':
                self.train_loader_iter = None
            elif split == 'val':
                self.val_loader_iter = None
            else:
                raise ValueError(f"Unknown split: {split}")
            return self.get_batch(split)

    def get_data_iter(self, split):
        data_loader = self.get_data_loader(split)
        data_iter = iter(data_loader)
        if split == 'train':
            self.train_loader_iter = data_iter
        elif split == 'val':
            self.val_loader_iter = data_iter
        else:
            raise ValueError(f"Unknown split: {split}")
        return data_iter

    def get_data_loader(self, split):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        epoch = self.dataset_counters[split]
        self.dataset_counters[split] += 1

        dataset = self.get_dataset(epoch, split)
        batch_size = self.config.train.batch_size

        if self.config.run.parallel_mode == 'ddp':
            sampler = DistributedSampler(dataset)
            sampler.set_epoch(epoch)
            loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)

        else:
            loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

        return loader

    def get_dataset(self, epoch, split):
        import glob
        from bert.dataset import BERTDatasetPrecached

        if split == 'train':
            pattern = self.config.train.dataset_pattern
            percentage = self.config.train.dataset_percentage
        elif split == 'val':
            pattern = self.config.train.val_dataset_pattern
            percentage = self.config.train.val_dataset_percentage
        else:
            raise ValueError(f"Unknown split: {split}")

        pattern = str(self.config.run.datasets_dir / pattern)
        # add an optional .gz extension to the pattern
        dataset_files = glob.glob(pattern) + glob.glob(pattern + '.gz')
        if len(dataset_files) == 0:
            raise ValueError(f"Dataset files not found with pattern {pattern}")
        dataset_files = sorted(dataset_files)
        dataset_file = dataset_files[epoch % len(dataset_files)]

        logging.info(f"Epoch: {epoch} - Loading dataset from {dataset_file}")

        dataset = BERTDatasetPrecached(dataset_file, percentage)
        return dataset
