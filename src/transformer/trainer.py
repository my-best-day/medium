import time
import torch
import logging
from pathlib import Path
from contextlib import nullcontext
from utils.timer import Timer


logger = logging.getLogger(__name__)

SKIP_ACCURACY = -1


class Trainer:
    def __init__(self, config, model, optimizer, task_handler):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.task_handler = task_handler

        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else \
            'float16'

        # initialize a GradScaler. If enabled=False scaler is a no-op
        enable_grad_scaler = torch.cuda.is_available() and dtype == 'float16'
        self.scaler = torch.amp.GradScaler(enabled=enable_grad_scaler)

        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        # 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        device_type = self.config.run.device if isinstance(self.config.run.device, str) else \
            self.config.run.device.type
        # note: float16 data type will automatically use a GradScaler
        pt_dtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[dtype]
        self.autocast_ctx = nullcontext() if device_type == 'cpu' else \
            torch.amp.autocast(device_type=device_type, dtype=pt_dtype)

        self.train_loader_iter = None
        self.val_loader_iter = None

        self.dataset_counters = {'train': 0, 'val': 0}

        # todo: take from config
        self.start_iter = 0
        self.iter = self.start_iter

        # switching from iter to sample count
        self.sample_iter = 0
        self.sample_iter_start = self.sample_iter

        is_ddp = self.config.run.parallel_mode == 'ddp'
        self.world_size = torch.distributed.get_world_size() if is_ddp else 1

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
                logits, loss = self.training_forward_and_loss(micro_step, X, Y)  # NOSONAR
                X, Y = self.get_batch('train')
                loss /= self.micro_step_count
                accumulated_loss += loss
                self.backward(loss)
                self.sample_iter += X.size(0) * self.world_size
            # outside micro-step
            logger.debug(f"loss: {accumulated_loss}")
            losses.append(accumulated_loss)
            self.step()
            self.optimizer.zero_grad()
            self.iter += 1

    def should_continue_training(self):
        if Path('./stop').exists() or Path('./stop_now').exists():
            logger.info("Stopping training because file './stop' or './stop_now' exists.")
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

    def training_forward_and_loss(self, micro_step, X, Y):  # NOSONAR X, Y denote batch tensors
        # in ddp, inhibit sync for all but the last micro-step
        sync_ctx = self.get_sync_context(micro_step)

        with self.autocast_ctx, sync_ctx:
            logits, loss = self.forward_and_loss(X, Y)
        return logits, loss

    def forward_and_loss(self, X, Y):  # NOSONAR X, Y denote batch tensors
        logits = self.model(X)
        loss = self.task_handler.get_loss(logits, Y)
        return logits, loss

    def get_sync_context(self, micro_step):
        """
        Returns a context manager that inhibits gradient synchronization if we are using
        DDP / the model supports that, AND, this is not last micro-step.
        """
        sync_context = None

        # check if the model supports no_sync (typically if using DDP)
        can_inhibit_syncing = callable(getattr(self.model, 'no_sync', None))

        # check if this is not the last micro-step
        is_not_last_micro_step = (micro_step != self.micro_step_count - 1)

        if can_inhibit_syncing and is_not_last_micro_step:
            sync_context = self.model.no_sync()
        else:
            sync_context = nullcontext()

        return sync_context

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self):
        if self.grad_clip > 0.0:
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
        ]
        if val_accuracy != SKIP_ACCURACY:
            items.append(f'v.acc: {val_accuracy:5.2%}')

        msg = ' | '.join(items)

        logger.info(msg)

    def log_tensorboard(self, train_loss, val_loss, val_accuracy, lr):
        if train_loss is not None:
            self.writer.add_scalar('train_loss', train_loss, self.iter)
        self.writer.add_scalar('val_loss', val_loss, self.iter)
        if val_accuracy != SKIP_ACCURACY:
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

        checkpoint = self.task_handler.gen_checkpoint(
            (self.model.module if is_wrapped else self.model),
            self.optimizer,
            iter,
            self.sample_iter,
            val_loss
        )

        torch.save(checkpoint, checkpoint_path)

    def is_model_wrapped(self):
        result = self.config.run.parallel_mode in ('dp', 'ddp')
        return result

    @torch.no_grad()
    def estimate_val_loss(self) -> tuple[float, float]:
        """
        Estimate the validation loss, and accuracy by running the model on the validation set.

        Returns:
            val_loss: The estimated validation loss
            val_accuracy: The estimated validation accuracy if applicable, otherwise SKIP_ACCURACY
        """
        losses = []
        total = 0
        correct = 0

        self.model.eval()

        dump_sentences = True
        for _ in range(self.val_iters):
            X, Y = self.get_batch('val')
            logits, loss = self.forward_and_loss(X, Y)
            losses.append(loss)

            if dump_sentences:
                self.task_handler.illustrate_predictions(X, Y, logits)
                dump_sentences = False

            probabilities = torch.softmax(logits, dim=-1)
            _, predicted = torch.max(probabilities, dim=-1)

            sample_total, sample_correct = self.task_handler.estimate_accuracy(Y, predicted)
            if sample_total is not None:
                total += sample_total
            if sample_correct is not None:
                correct += sample_correct

        self.model.train()

        val_loss = sum(losses) / len(losses)
        val_loss = self.sync_up(val_loss)
        if correct == SKIP_ACCURACY:
            val_accuracy = SKIP_ACCURACY
        else:
            val_accuracy = correct / total
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

    def get_batch(self, split):
        if split == 'train':
            iterator = self.train_loader_iter
        elif split == 'val':
            iterator = self.val_loader_iter
        else:
            raise ValueError(f"Unknown split: {split}")

        if iterator is None:
            iterator = self.get_data_iter(split)

        try:
            X, Y = next(iterator)
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
            sampler = DistributedSampler(dataset, shuffle=True)
            sampler.set_epoch(epoch)
            loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                                pin_memory=self.config.run.async_to_device)
        else:
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True,
                                pin_memory=self.config.run.async_to_device)

        return loader

    def get_dataset(self, epoch, split):
        return self.task_handler.get_dataset(epoch, split)
