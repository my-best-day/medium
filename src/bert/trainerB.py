import time
import torch
import logging
from pathlib import Path
from contextlib import nullcontext

from bert.timer import Timer


# TODO: adjust eval_interval, max_iters, val_iters by world_size (number of GPUs)


class TrainerB:
    def __init__(self, config, model, optimizer, tokenizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else \
            'float16'

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        # 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        device_type = self.config.run.device if isinstance(self.config.run.device, str) else \
            self.config.run.device.type
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[dtype]
        self.autocast_ctx = nullcontext() if device_type == 'cpu' else \
            torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.train_loader_iter = None
        self.val_loader_iter = None

        self.dataset_counters = {'train': 0, 'val': 0}

        # todo: take from config
        self.start_iter = 0
        self.iter = self.start_iter
        self.micro_step_count = 1
        self.grad_clip = 1.0
        self.val_iters = 10
        self.best_val_loss = float('inf')

    def train(self):
        timer = Timer()
        losses = []
        X, Y = self.get_batch('train')
        while self.should_continue_training():
            lr = self.adjust_lr()
            if self.should_estimate_loss():
                elapsed = timer.elapsed(restart=False)
                self.estimate_loss_and_log_progress(elapsed, losses, lr)
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
        if Path('./stop').exists() or Path('./stop_now').exists():
            logging.info("Stopping training because file './stop' or './stop_now' exists.")
            result = False
        else:
            result = self.iter < self.config.train.max_iters
        return result

    def should_estimate_loss(self):
        iters = self.iter - self.start_iter
        result = iters % self.config.train.val_interval == 0
        return result

    def adjust_lr(self):
        lr = self.get_lr(self.iter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self, iter):
        import math

        lr = self.config.train.learning_rate
        min_lr = self.config.train.min_learning_rate

        warmup_iters = self.config.train.warmup_iters
        lr_decay_iters = self.config.train.lr_decay_iters

        if iter < warmup_iters:
            lr = iter * lr / warmup_iters

        elif iter <= lr_decay_iters:
            ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # range 0..1
            lr = min_lr + coeff * (lr - min_lr)

        else:  # it > lr_decay_iters:
            lr = min_lr

        return lr

    def forward_and_loss(self, micro_step, X, Y):
        no_sync_ctx = self.model.no_sync() if (micro_step != self.micro_step_count - 1) else \
            nullcontext()
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

    def estimate_loss_and_log_progress(self, elapsed, train_losses, lr):
        train_loss = sum(train_losses) / len(train_losses) if len(train_losses) > 0 else None
        val_loss, val_accuracy = self.estimate_val_loss()
        if self.config.run.is_primary:
            self.log_progress(elapsed, train_loss, val_loss, val_accuracy, lr)

            if val_loss < self.best_val_loss and self.iter > self.config.train.val_interval:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.iter, val_loss)

    def should_save_checkpoint(self, val_loss):
        if val_loss >= self.best_val_loss:
            return False
        iters = self.iter - self.start_iter
        if iters < self.val_iters:
            return False
        return True

    def log_progress(self, elapsed, train_loss, val_loss, val_accuracy, lr):
        self.log_tensorboard(train_loss, val_loss, val_accuracy, lr)

        if self.config.run.wandb:
            self.log_wandb(train_loss, val_loss, val_accuracy, lr)

        num_digits = len(f'{self.config.train.max_iters:,}')
        per_iteration = (elapsed / self.iter) if self.iter > 0 else 0.5
        remaining = per_iteration * (self.config.train.max_iters - self.iter)
        loss_str = 'None ' if train_loss is None else f'{train_loss:5.2f}'
        items = [
            f'{self.iter / self.config.train.max_iters:4.0%}',
            f'{self.iter:>{num_digits},}/{self.config.train.max_iters:,}',
            f'e: {time.strftime("%H:%M:%S", time.gmtime(elapsed))}',
            f'r: {time.strftime("%H:%M:%S", time.gmtime(remaining))}',
            f'{time.strftime("%M:%S", time.gmtime(per_iteration * 1000.0))} /Kit',
            f'lr: {lr * 1000:6.4f} e-3',
            f't.loss: {loss_str}',
            f'v.loss: {val_loss:5.2f}',
            f'v.accu: {val_accuracy:.1%}',
        ]
        msg = ' | '.join(items)
        logging.info(msg)

    def log_tensorboard(self, train_loss, val_loss, val_accuracy, lr):
        if train_loss is not None:
            self.writer.add_scalar('train_loss', train_loss, self.iter)
        self.writer.add_scalar('val_loss', val_loss, self.iter)
        self.writer.add_scalar('val_accuracy', val_accuracy, self.iter)
        self.writer.add_scalar('lr', lr, self.iter)

    def log_wandb(self, train_loss, val_loss, val_accuracy, lr):
        import wandb
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'lr': lr},
            step=self.iter
        )

    def save_checkpoint(self, iter: int, val_loss: float):
        # skip checkpoint if this is not the main process
        if not self.config.run.is_primary:
            return

        is_wrapped = self.is_model_wrapped()

        name = "checkpoint.pt"
        checkpoint_path = self.config.run.checkpoints_dir / name

        torch.save(
            {
                'format': 'bert1',
                'version': 1.0,
                'iter': iter,
                'model': (self.model.module if is_wrapped else self.model).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'config': self.config.to_dict(),
            },
            checkpoint_path
        )

    def is_model_wrapped(self):
        result = self.config.run.parallel_mode in ('dp', 'ddp')
        return result

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
            X = X.to(self.config.run.device, non_blocking=self.config.run.async_to_device)
            Y = Y.to(self.config.run.device, non_blocking=self.config.run.async_to_device)
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
            loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                                pin_memory=self.config.run.async_to_device)
        else:
            loader = DataLoader(dataset, batch_size=batch_size,
                                pin_memory=self.config.run.async_to_device)

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

        logging.debug(f"Epoch: {epoch} - Loading dataset from {dataset_file}")

        dataset = BERTDatasetPrecached(dataset_file, percentage)
        return dataset
