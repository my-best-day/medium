import logging

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = 0.00025
        self.factor = 5.0 / 6.0

    def step(self, epoch):
        if epoch < 5:
            self.lr /= self.factor
        else:
            self.lr *= self.factor

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        logging.info(f'Epoch {epoch} learning rate: {self.lr}')