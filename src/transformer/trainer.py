import time
import math
import torch
import logging
from collections import deque
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

        self.configure_precision()

        self.loader_iter_map: dict[str, iter] = {}

        self.dataset_counters = {}

        # todo: take from config
        self.start_iter = 0
        self.iters = self.start_iter

        # switching from iter to sample count
        self.sample_iter = 0
        self.sample_iter_start = self.sample_iter

        is_ddp = self.config.run.parallel_mode == 'ddp'
        self.world_size = torch.distributed.get_world_size() if is_ddp else 1

        self.micro_step_count = 1
        self.grad_clip = 1.0
        self.split_iters = {
            'val': self.config.train.val_iters,
            'test': self.config.train.test_iters
        }

        val_loss_window_size = int(max(3, 0.00025 * config.train.max_iters))
        # 25_000 iters -> 25 / 25_000 = 0.001; 100_000 iters -> 25 / 100_000 = 0.00025
        self.min_improvement = 25 / config.train.max_iters
        self.last_n_val_losses = deque(maxlen=val_loss_window_size)
        self.resume_lr = None
        self.best_val_loss = float('inf')

    def configure_precision(self):
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

    def train(self):
        timer = Timer()
        losses = []
        logger.info(">>> >>> Starting training")
        X, Y = self.get_batch('train', True)
        logger.info(">>> >>> Got batch")
        while self.should_continue_looping(self.config.train.max_iters, self.iters):
            self.set_micro_step_count()
            micro_step_adj = self.get_micro_step_lr_adjustment()
            lr = self.adjust_lr(micro_step_adj)
            if self.should_estimate_loss():
                elapsed = timer.elapsed(restart=False)
                self.estimate_loss_and_log_progress(elapsed, losses, lr)
                losses = []
            elif self.iters % 10 == 0 and len(losses) > 0:
                train_loss = self.average_synced_up_loss(losses)
                self.log_progress(timer.elapsed(restart=False), train_loss, None, SKIP_ACCURACY, lr)
            # enter micro-step
            accumulated_loss = 0.0
            for micro_step in range(self.micro_step_count):
                logits, loss = self.training_forward_and_loss(micro_step, X, Y)  # NOSONAR
                X, Y = self.get_batch('train', True)
                loss /= self.micro_step_count
                accumulated_loss += loss
                self.backward(loss)
                self.sample_iter += X.size(0) * self.world_size
            # outside micro-step
            logger.debug(f"loss: {accumulated_loss}")
            losses.append(accumulated_loss)
            self.step()
            self.optimizer.zero_grad()
            self.iters += 1

    def set_micro_step_count(self):
        """
        Current implementation sets micro-step counts to 1 for two cycles
        then to 2 for one cycle. We use val_interval to determine the cycle length.
        (we estimate the val loss every val_interval iterations)
        """
        flip_every = self.config.train.val_interval
        self.micro_step_count = 1 + (self.iters // flip_every) % 3

    def get_micro_step_lr_adjustment(self):
        """
        micro-step-count = 1 => adjustment = 0.05
        micro-step-count = 2 => adjustment = -0.15
        the adjustment shrinks by decreasing amount as micro-step-count increases
        micro-step-count <= 8 => adjustment = -0.15 ** (2 / micro-step-count)

        we take this adjustment and use it to adjust a ramp function that affects the learning
        rate a long one cycle (val_interval iterations).
        The adjustment starts with zero, increase linearly to the adjustment value
        in the middle of the cycle, then decrease linearly back to zero.

        The goal is: decrease the learning rate the the micro-step count increases, and
        do this is gradually, using the ramp function, to avoid sudden change to the
        learning rate.
        """
        micro_step_count = self.micro_step_count
        adjustment_unit = 0.15  # percent
        if micro_step_count == 1:
            lr_adjustment = adjustment_unit / 3
        elif micro_step_count == 2:
            lr_adjustment = -adjustment_unit
        elif micro_step_count <= 8:
            lr_adjustment = -adjustment_unit ** (2 / micro_step_count)
        else:
            raise ValueError(f"Invalid micro_step_count: {micro_step_count}")
        return lr_adjustment

    @torch.no_grad()
    def test(self):
        """
        Run the model on the test set and log the loss and accuracy.
        """
        loss, accuracy = self.estimate_loss('test', False)
        accuracy = 0 if accuracy == SKIP_ACCURACY else accuracy
        logger.info(f"Test loss: {loss:5.2f}, accuracy: {accuracy:4.3%}")

    def should_continue_looping(self, max_iters: int, iters: int) -> bool:
        if Path('./stop').exists() or Path('./stop_now').exists():
            logger.info("Stopping training because file './stop' or './stop_now' exists.")
            result = False
        elif max_iters is not None:
            result = iters < max_iters
        else:
            result = True
        return result

    def should_estimate_loss(self):
        iters = self.iters - self.start_iter
        result = iters % self.config.train.val_interval == 0 and iters > 0
        return result

    def adjust_lr(self, micro_step_adj):
        lr = self.get_lr(micro_step_adj)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self, micro_step_adj):
        """
        Get the learning rate for the current iteration.

        The learning rate is adjusted in the following steps:

        1. Cosine Annealing with Warmup:
            - The learning rate starts from 0 and increases linearly to the base learning rate
              during the warmup period.
            - After the warmup period, it follows a cosine annealing schedule until
              `lr_decay_iters`.

        2. Warmup from Checkpoint:
            - If resuming from a checkpoint, the learning rate is gradually increased from the
              checkpoint's learning rate to the current learning rate over `resume_warmup_iters`.

        3. Learning Rate Adjustment:
            - The adjustments are applied in two ways:
                3.a. Ramp Function:
                    - Along the learning rate cycle (`val_interval`), the learning rate is
                      gradually nudged up and then gradually down.
                3.b. Micro-Step Count Adjustment:
                    - The adjustment of the ramp function is determined by the micro-step count:
                    - Micro-step count 1: slight adjustment upwards (up to 5%).
                    - Micro-step count 2: adjustment downwards (up to 15%).
                    - Micro-step counts 3 to 8: adjustment downwards with a gently increasing
                      amount.
        """
        lr = self.get_lr_cosine_annealing_with_warmup()
        lr = self.get_smoothed_lr_after_resume(lr)
        # introduces a ramp adjustment sensitive to the micro-step count
        adj = self.get_cyclical_lr_adjustment(micro_step_adj)
        lr = (1 + adj) * lr
        return lr

    def get_micro_step_adjusted_lr_adjustment(self, lr):
        amount = 0.15 * lr
        adjustment = self.get_cyclical_lr_adjustment(amount)
        if self.micro_step_count == 1:
            # standard
            pass
        elif self.micro_step_count == 2:
            adjustment = - adjustment
        elif self.micro_step_count <= 8:
            adjustment = - adjustment
        else:
            raise ValueError(f"Invalid micro_step_count: {self.micro_step_count}")
        return adjustment

    def get_cyclical_lr_adjustment(self, multiplier=1.0):
        cycle_len = self.config.train.val_interval
        iters = self.iters
        return self.cyclical_ramp(cycle_len, multiplier, iters)

    @staticmethod
    def cyclical_ramp(cycle_len, multiplier, iters):
        """
        Adjust the learning rate for the current micro-step.
        """
        position = iters % cycle_len
        ramp = 1 - abs(2 * position / cycle_len - 1)
        result = multiplier * ramp
        return result

    def get_smoothed_lr_after_resume(self, lr):
        """
        Gradually warm up the learning rate from the resume value.
        """
        resume_lr = self.resume_lr
        resume_warmup_iters = self.config.train.resume_warmup_iters
        if resume_lr is not None:
            rel_iters = self.iters - self.start_iter
            if rel_iters < resume_warmup_iters:
                # smoothly increase learning rate from the resume value
                lr = resume_lr + (lr - resume_lr) * rel_iters / resume_warmup_iters
        return lr

    def get_lr_cosine_annealing_with_warmup(self):
        """
        Standard cosine annealing with warmup learning rate scheduler.
        """
        import math

        iters = self.iters

        lr = self.config.train.learning_rate
        min_lr = self.config.train.min_learning_rate

        warmup_iters = self.config.train.warmup_iters
        lr_decay_iters = self.config.train.lr_decay_iters

        if iters < warmup_iters:
            lr = iters * lr / warmup_iters

        elif iters <= lr_decay_iters:
            ratio = (iters - warmup_iters) / (lr_decay_iters - warmup_iters)
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

    def estimate_loss_and_log_progress(self, elapsed, train_losses, lr, estimate_loss=True):
        train_loss = self.average_synced_up_loss(train_losses)

        if estimate_loss:
            val_loss, val_accuracy = self.estimate_loss('val', True)
            self.last_n_val_losses.append(val_loss)
            avg_val_loss = sum(self.last_n_val_losses) / len(self.last_n_val_losses)
        else:
            val_loss = None
            val_accuracy = SKIP_ACCURACY

        self.log_progress(elapsed, train_loss, val_loss, val_accuracy, lr)

        if self.config.run.is_primary and estimate_loss:
            if self.should_save_checkpoint(avg_val_loss):
                self.save_checkpoint(self.iters, avg_val_loss, lr)
                self.best_val_loss = avg_val_loss

    def should_save_checkpoint(self, avg_val_loss):
        # allow the user to save checkpoint on demand
        if Path('./save_checkpoint').exists():
            Path('./save_checkpoint').unlink()
            return True
        if self.iters < self.config.train.val_interval:
            return False
        if (self.best_val_loss - avg_val_loss) < self.min_improvement:
            return False
        return True

    def log_progress(self, elapsed, train_loss, val_loss, val_accuracy, lr):
        if not self.config.run.is_primary:
            return

        self.log_tensorboard(train_loss, val_loss, val_accuracy, lr)

        if self.config.run.wandb:
            self.log_wandb(train_loss, val_loss, val_accuracy, lr)

        num_digits = len(f'{self.config.train.max_iters:,}')
        per_iteration = (elapsed / self.iters) if self.iters > 0 else 0.5
        remaining = per_iteration * (self.config.train.max_iters - self.iters)
        t_loss_str = 'None ' if train_loss is None else f'{train_loss:5.2f}'
        v_loss_str = 'None ' if val_loss is None else f'{val_loss:5.2f}'
        relative_iters = self.iters - self.start_iter
        items = [
            f'{self.iters / self.config.train.max_iters:4.0%}',
            f'{self.iters:>{num_digits},}/{self.config.train.max_iters:,} ({relative_iters:,})',
            f'e: {time.strftime("%H:%M", time.gmtime(elapsed))}',
            f'r: {time.strftime("%H:%M", time.gmtime(remaining))}',
            f'{time.strftime("%M:%S", time.gmtime(per_iteration * 1000.0))} /Kit',
            f'lr: {lr * 1000:6.4f} e-3',
            f't.loss: {t_loss_str}',
            f'v.loss: {v_loss_str}',
            f'micro-steps: {self.micro_step_count}',
        ]
        if val_accuracy != SKIP_ACCURACY:
            items.append(f'v.acc: {val_accuracy:5.2%}')

        msg = ' | '.join(items)

        logger.info(msg)

    def log_tensorboard(self, train_loss, val_loss, val_accuracy, lr):
        if train_loss is not None:
            self.writer.add_scalar('train_loss', train_loss, self.iters)
        if val_loss is not None:
            self.writer.add_scalar('val_loss', val_loss, self.iters)
        if val_accuracy != SKIP_ACCURACY:
            self.writer.add_scalar('val_accuracy', val_accuracy, self.iters)
        self.writer.add_scalar('lr', lr, self.iters)
        self.writer.add_scalar('micro_steps', self.micro_step_count, self.iters)

    def log_wandb(self, train_loss, val_loss, val_accuracy, lr):
        import wandb
        wandb.log({
            'step': self.iters,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'lr': lr,
            'micro_steps': self.micro_step_count,
        })

    def save_checkpoint(self, iter: int, val_loss: float, lr: float):
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
            val_loss,
            lr
        )

        torch.save(checkpoint, checkpoint_path)

    def is_model_wrapped(self):
        result = self.config.run.parallel_mode in ('dp', 'ddp')
        return result

    @torch.no_grad()
    def estimate_loss(self, split, illustrate) -> tuple[float, float]:
        """
        Estimate the validation/test loss, and accuracy by running the model on the
        validation/test set.

        Returns:
            val_loss: The estimated validation/test loss
            val_accuracy: The estimated validation accuracy if applicable, otherwise SKIP_ACCURACY
        """
        # consider averaging the losses by sample count rather than by iteration to
        # account for partial batches. Very low priority, we average lots of full batches
        # and only very few expected to be partial.
        losses = []
        total = 0
        correct = 0

        max_iters = self.split_iters.get(split, None)

        self.model.eval()
        try:
            should_illustrate = illustrate
            iters = 0
            while self.should_continue_looping(max_iters, iters):
                X, Y = self.get_batch(split, True)
                if X is None or Y is None:
                    break

                logits, loss = self.forward_and_loss(X, Y)
                losses.append(loss)

                if should_illustrate:
                    self.task_handler.illustrate_predictions(self.model, X, Y, logits)
                    should_illustrate = False

                sample_total, sample_correct = self.task_handler.estimate_accuracy(Y, logits)
                total += sample_total
                correct += sample_correct

                self.log_estimate_progress(split, iters, loss.item(), correct, total)

                iters += 1

        finally:
            self.model.train()

        loss = self.average_synced_up_loss(losses)
        if total == 0:
            accuracy = SKIP_ACCURACY
        else:
            accuracy = self.ratio_synced_up_values(correct, total)
        return loss, accuracy

    def log_estimate_progress(self, split, iters, loss, correct, total):
        if self.config.run.is_primary and (iters % 100 == 0):
            logger.info("%s iteration: %s, loss: %5.2f, accuracy: %4.3f",
                        split, iters, loss, correct / total if total > 0 else 0)

    def average_synced_up_loss(self, losses):
        """
        Calculates the ratio of the sum of losses across processes to the sum of their counts
        """
        total_loss = sum(losses) if len(losses) > 0 else 0.0
        loss_count = len(losses)
        loss = self.ratio_synced_up_values(total_loss, loss_count)
        return loss

    def ratio_synced_up_values(self, nominator, denominator):
        """
        Calculates the ratio of the sums across all processes
        """
        nominator = self.sync_up(nominator)
        denominator = self.sync_up(denominator)
        ratio = nominator / denominator if denominator != 0 else 0.0
        return ratio

    def sync_up(self, item):
        """
        Sum up the item across all processes in the distributed setting
        """
        if self.config.run.parallel_mode == 'ddp':
            if isinstance(item, torch.Tensor):
                tensor = item
            else:
                tensor = torch.tensor(item).to(self.config.run.device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
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

    def get_batch(self, split, allow_reset):
        """
        Retrieves the next batch of data from the specified split (train, val, test).

        Args:
            split (str): The data split to retrieve the batch from ('train' or 'val', 'test').
            allow_reset (bool): If True, resets the iterator and starts a new epoch
                                when the current iterator is exhausted.
                                If False, returns (None, None) when the iterator is exhausted.

        Returns:
            tuple: A tuple (X, Y) containing the input data and corresponding labels,
                or (None, None) if the iterator is exhausted and `allow_reset` is False.
        """
        iterator = self.loader_iter_map.get(split, None)
        if iterator is None:
            iterator = self.gen_data_iter(split)

        try:
            X, Y = next(iterator)
            X = X.to(self.config.run.device, non_blocking=self.config.run.async_to_device)
            Y = Y.to(self.config.run.device, non_blocking=self.config.run.async_to_device)
            return X, Y
        except StopIteration:
            # Handle the case when the iterator is exhausted
            # For example, you can reset the iterator or raise an exception
            if allow_reset:
                self.loader_iter_map[split] = None
                return self.get_batch(split, allow_reset)
            else:
                return None, None

    def gen_data_iter(self, split):
        data_loader = self.get_data_loader(split)
        data_iter = iter(data_loader)
        self.loader_iter_map[split] = data_iter
        return data_iter

    def get_data_loader(self, split):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        epoch = self.dataset_counters.get(split, 0)
        self.dataset_counters[split] = epoch + 1

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
