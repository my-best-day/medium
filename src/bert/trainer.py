import glob
import time
import torch
import datetime
from pathlib import Path
from bert.bert import BERT
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from bert.dump_sentences import DumpStentences

from mtimer import MTimer
from bert.timer import Timer
from bert.dataset import BERTDatasetPrecached

class BERTTrainer:
    def __init__(self,
                 model: BERT,
                 logs_dir: Path,
                 checkpoints_dir: Path,
                 print_every: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 tokenizer,
                 device: str,
                 ):
        self.model = model
        self.print_every = print_every
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.tokenizer = tokenizer

        self.id = None

        self.dump_sentences = DumpStentences(tokenizer)

        betas = (0.9, 0.999)
        weight_decay = 0.015

        self.criterion = torch.nn.NLLLoss(ignore_index=0).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=learning_rate, betas=betas) # , weight_decay=weight_decay)
        # self.optimizer_schedule = ScheduledOptim(self.optimizer, model.d_model, n_warmup_steps=10000)

        self.logs_dir = logs_dir
        self.checkpoints_dir = checkpoints_dir
        self._writer = None

        self.start_epoch = 0
        self.epoch = 0

        # for p in self.model.parameters():
        #     print(p, p.nelement())

        total_parameters = sum([p.nelement() for p in self.model.parameters()])
        print(f"Total Parameters: {total_parameters:,}")

    @property
    def writer(self):
        if self._writer is None:
            self._writer = SummaryWriter(str(self.logs_dir))
        return self._writer

    def before_epoch(self, epoch):
        pass

    def train(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            loss = self.train_epoch(self.epoch)
            self.save_checkpoint(self.epoch + 1, -1, loss)



    def eval_loss(self, loader):
        timer = Timer()
        mtimer = MTimer()
        losses = []
        with torch.no_grad():
            self.model.eval()
            mtimer.start('batch')
            for i, data in enumerate(loader):
                mtimer.end('batch')
                mtimer.start("device")
                sentence, labels = data
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                mtimer.end("device")
                mtimer.start("forward")
                mlm_out = self.model(sentence)
                mtimer.end("forward")
                mtimer.start("loss")
                loss = self.criterion(mlm_out.transpose(1, 2), labels)
                mtimer.end("loss")
                losses.append(loss.item())
                mtimer.start('batch')
            self.model.train()
        loss = sum(losses) / len(losses)
        timer.print(f"Eval loss: {loss:6.2f}, {len(losses)} batches")
        mtimer.dump()
        return loss

    def train_epoch(self, epoch):
        print(f"Begin epoch {epoch}")

        loader, eval_loader = self.before_epoch(epoch)

        start_time = time.time()
        mtimer = MTimer()
        accumulated_loss = 0
        timer = Timer()
        mtimer.start('batch')
        for i, data in enumerate(loader):
            mtimer.end('batch')

            mtimer.start('device')
            sentence, labels = data
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            mtimer.end('device')

            mtimer.start('forward')
            mlm_out = self.model(sentence)
            mtimer.end('forward')

            if (i + 1) % self.print_every == 0:
                # import numpy as np
                # np.set_printoptions(formatter={'float': '{:0.2f}'.format})
                # print(mlm_out.detach().cpu().numpy()[0,0,:])

                print("=" * 70 )
                predicted = self.dump_sentences.batched_debug(sentence, labels, mlm_out)
                print("\n".join(predicted[:5]))
                print("=" * 70 )

            mtimer.start('loss')
            loss = self.criterion(mlm_out.transpose(1, 2), labels)
            accumulated_loss += loss
            mtimer.end('loss')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if (i + 1) % self.print_every == 0:
                elapsed = time.time() - start_time
                mtimer.start('eval')
                eval_loss = self.eval_loss(eval_loader)
                mtimer.end('eval')
                summary = self.training_summary(elapsed, (i+1), accumulated_loss, eval_loss)
                print(summary)
                # self.save_checkpoint(self.epoch, i, loss)
                accumulated_loss = 0
                timer.print("train step", restart=True)

            mtimer.start('batch')

        eval_loss = self.eval_loss(eval_loader)
        summary = summary.self.training_summary(elapsed, (i+1), accumulated_loss)
        print(summary)
        mtimer.dump()

        return loss

    def training_summary(self, elsapsed, index, accumulated_loss, eval_loss=None):
        passed = index / self.batch_count
        global_step = self.epoch * self.batch_count + index
        mlm_loss = accumulated_loss / self.print_every
        items = [
            time.strftime('%H:%M:%S', time.gmtime(elsapsed)),
            f"(r:{self.estimate_remaining_time(passed, elsapsed)})",
            f"Epocn {self.epoch}",
            f"{index} / {self.batch_count} ({passed:6.2%})",
            f"MLM loss: {mlm_loss:6.2f}",
        ]
        if eval_loss is not None:
            items.append(f"Eval loss: {eval_loss:6.2f}")
        result = " | ".join(items)
        self.writer.add_scalar("MLM loss", mlm_loss, global_step=global_step)
        return result

    @staticmethod
    def estimate_remaining_time(passed: float, elapsed: float):
        if passed <= 0:
            return "00:00:00"
        remaining = elapsed / passed - elapsed
        formatted = time.strftime('%H:%M:%S', time.gmtime(remaining))
        return formatted


    def save_checkpoint(self, epoch: int, index: int, loss: float):
        global_step = epoch * self.batch_count + index
        start_time = time.time()
        name = f"bert_epoch{epoch}_index{global_step}_{datetime.datetime.utcnow().timestamp():.0f}.pt"
        torch.save({
            'version': 0.1,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, self.checkpoints_dir / name)

        text = "\n".join([
            "",
            "=" * 70,
            f"Model saved as '{name}' for {time.time() - start_time:.2f} secs",
            "=" * 70,
            ""
        ])
        print(text)


    def load_checkpoint(self, path: Path):
        print("=" * 70)
        print(f"Restoring model {path}")
        checkpoint = torch.load(path)
        version = checkpoint.get('version', 0)
        self.epoch = checkpoint['epoch'] + (1 if version == 0 else 0)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = self.epoch
        print("Model is restored.")
        print("=" * 70)


class BERTTrainerSingleDataset(BERTTrainer):

    def __init__(self,
                 model: BERT,
                 logs_dir: Path,
                 checkpoints_dir: Path,
                 print_every: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 tokenizer,
                 device: str,
                 dataset: Dataset = None):
            self.dataset = dataset
            self.loader = None
            super().__init__(model, logs_dir, checkpoints_dir, print_every, batch_size, learning_rate, epochs, tokenizer, device)

    def before_epoch(self, epoch):
        if self.loader is None:
            self.ds_size = len(self.dataset)
            self.batch_size = self.batch_size
            self.batch_count = self.ds_size // self.batch_size

            # self.loader = DataLoader(dataset, batch_size, num_workers=1, shuffle=True, pin_memory=True)
            self.loader = DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        return self.loader


class BERTTrainerPreprocessedDatasets(BERTTrainer):
    def __init__(self,
                 model: BERT,
                 logs_dir: Path,
                 checkpoints_dir: Path,
                 print_every: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 tokenizer,
                 device: str,
                 dataset_dir: Path,
                 dataset_pattern: str,
                 eval_pattern: str,
            ):
        self.dataset_dir = dataset_dir
        self.dataset_pattern = dataset_pattern
        self.eval_pattern = eval_pattern
        super().__init__(model, logs_dir, checkpoints_dir, print_every, batch_size, learning_rate, epochs, tokenizer, device)


    def before_epoch(self, epoch):
        dataset = self.get_dataset(epoch, True)
        self.ds_size = len(dataset)
        self.batch_size = self.batch_size
        self.batch_count = self.ds_size // self.batch_size
        loader = DataLoader(dataset, self.batch_size, shuffle=True, pin_memory=True)

        eval_dataset = self.get_dataset(epoch, False)
        eval_loader = DataLoader(eval_dataset, self.batch_size, shuffle=False, pin_memory=True)
        return loader, eval_loader

    def get_dataset(self, epoch, train):
        pattern = self.dataset_pattern if train else self.eval_pattern

        pattern = str(self.dataset_dir / pattern)
        # add an optional .gz extension to the pattern
        dataset_files = glob.glob(pattern) + glob.glob(pattern + '.gz')
        if len(dataset_files) == 0:
            raise ValueError(f"Dataset files not found with pattern {pattern}")
        dataset_files = sorted(dataset_files)
        dataset_file = dataset_files[epoch % len(dataset_files)]

        print(f"Epoch: {epoch} - Loading dataset from {dataset_file}")

        dataset = BERTDatasetPrecached(dataset_file)
        return dataset
