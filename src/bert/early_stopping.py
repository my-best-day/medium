# WIP, not used or tested yet
import torch


class EarlyStopping:
    def __init__(self, model, path, patience, verbose, delta):
        self.model = model
        self.path = path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.delta = delta
        self.stop_flag = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss >= self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f"Early Stopping, loss doesn't improves. Current: {loss}, "
                      f"Best: {self.best_loss}")
                self.stop_flag = True
        elif loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
            self.save_checkpoint(loss)

    def save_checkpoint(self, loss):
        if self.verbose:
            print(f'Loss decreased {self.best_loss} --> {loss}. Saving model ...')
        state = {
            'model': self.model.state_dict(),
            'loss': loss,
            'counter': self.counter
        }
        torch.save(state, self.path + 'checkpoints/' + 'checkpoint.pth')
