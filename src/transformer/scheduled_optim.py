# WIP, not used or tested yet
import torch
import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def foo(optimizer, warmup_iters, lr_decay_iters, params, config, stages):
    optimizer = torch.optim.AdamW(params, lr=config.train.learning_rate)

    # Define warmup phase
    warmup_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda iter: iter / warmup_iters if iter < warmup_iters else 1.0
    )

    # Cosine annealing phase
    cosine_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=lr_decay_iters - warmup_iters,
        eta_min=config.train.min_learning_rate
    )

    if stages == 2:
        # Define a combined scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, cosine_lr_scheduler],
            milestones=[warmup_iters]
        )
    elif stages == 3:
        # If you want to stop decaying at the minimum learning rate after decay_iters,
        # you can add a third phase:
        constant_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda iter: config.train.min_learning_rate / optimizer.param_groups[0]['lr']
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, cosine_lr_scheduler, constant_lr_scheduler],
            milestones=[warmup_iters, lr_decay_iters]
        )
    else:
        raise ValueError(f"Invalid number of stages: {stages}")

    return scheduler


def get_lr_scheduler(optimizer, config):
    stages = config.train.stages
    warmup_iters = config.train.warmup_iters
    lr_decay_iters = config.train.lr_decay_iters
    params = config.train.params
    return foo(optimizer, warmup_iters, lr_decay_iters, params, config, stages)
