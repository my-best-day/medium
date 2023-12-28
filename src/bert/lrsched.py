class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = 0.00025
        self.factor = 0.8

    def step(self, epoch):
        self.lr *= self.factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        print(f'Epoch {epoch} learning rate: {self.lr}')