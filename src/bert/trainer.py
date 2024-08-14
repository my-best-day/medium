import time
from venv import logger
import torch
import logging
from pathlib import Path
from contextlib import nullcontext

from bert.timer import Timer
from bert.dump_sentences import DumpStentences

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, model, optimizer, tokenizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.dumper = DumpStentences(tokenizer)

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
                logits, loss = self.forward_and_loss(micro_step, X, Y)  # NOSONAR
                X, Y = self.get_batch('train')
                loss /= self.micro_step_count
                accumulated_loss += loss
                self.backward(loss)
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

    def forward_and_loss(self, micro_step, X, Y):  # NOSONAR upper case X, Y denote batch tensors
        # in ddp, inhibit sync for all but the last micro-step
        sync_ctx = self.get_sync_context(micro_step)

        with self.autocast_ctx, sync_ctx:
            logits = self.model(X)
            if self.config.model.task_type == 'mlm':
                loss_logits = logits.transpose(1, 2)
                loss = torch.nn.functional.cross_entropy(loss_logits, Y, ignore_index=0)
            else:
                loss_logits = logits
            loss = torch.nn.functional.cross_entropy(loss_logits, Y)
            # logger.info("v logits shape: %s, Y shape: %s", loss_logits.shape, Y.shape)
            # logger.info("v logits: %s", loss_logits)
            # logger.info("v Y: %s", Y)

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
            f'v.accu: {val_accuracy:.1%}',
        ]
        msg = ' | '.join(items)
        logger.info(msg)

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

        dump_sentences = self.config.model.task_type == 'mlm'
        for _ in range(self.val_iters):
            X, Y = self.get_batch('val')
            logits = self.model(X)

            if self.config.model.task_type == 'mlm':
                # For MLM tasks
                loss_logits = logits.transpose(1, 2)  # Shape: [batch_size, vocab_size, seq_len]
                loss = torch.nn.functional.cross_entropy(loss_logits, Y, ignore_index=0)
            else:
                # For classification tasks like CoLA
                loss_logits = logits  # Shape: [batch_size, num_classes]
                # logger.info("t logits shape: %s, Y shape: %s", loss_logits.shape, Y.shape)
                loss = torch.nn.functional.cross_entropy(loss_logits, Y)  # Y should be [batch_size]
            # logger.info("t logits: %s", logits)
            # logger.info("t Y: %s", Y)

            losses.append(loss)

            if dump_sentences:
                dump_sentences = False
                debug_text = self.dumper.batched_debug(X, Y, logits)
                print("\n".join(debug_text))

            probabilities = torch.softmax(logits, dim=-1)
            _, predicted = torch.max(probabilities, dim=-1)

            # flatten tensors for comparison
            y_flat = Y.view(-1)
            predicted_flat = predicted.view(-1)

            if self.config.model.task_type == 'mlm':
                # mask: ignore padding (assumed 0) and focus on masked tokens (assumed non zero)
                mask = (y_flat != 0)
                total += mask.sum().item()
                correct += (predicted_flat[mask] == y_flat[mask]).sum().item()
            else:
                total += Y.size(0)
                correct += (predicted == Y).sum().item()

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
        if self.config.model.task_type == 'mlm':
            dataset = self.get_mlm_dataset(epoch, split)
        elif self.config.model.task_type == 'cola':
            dataset = self.get_cola_dataset(split)
        else:
            raise ValueError(f"Unknown dataset: {self.config.run.dataset}")
        return dataset

    def get_mlm_dataset(self, epoch, split):
        import glob
        from bert.bert_mlm_dataset_precached import BertMlmDatasetPrecached

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

        logger.info(f"*** *** *** *** Epoch: {epoch} - Loading dataset from {dataset_file}")

        dataset = BertMlmDatasetPrecached(dataset_file, percentage)
        return dataset

    def get_cola_dataset(self, split):
        assert split in ('train', 'val')
        from bert.cola.cola_dataset import ColaDataset
        if split == 'train':
            filename = 'in_domain_train.tsv'
        elif split == 'val':
            filename = 'in_domain_dev.tsv'
        path = self.config.run.datasets_dir / filename
        dataset = ColaDataset(path, self.tokenizer, self.config.model.seq_len)
        return dataset
