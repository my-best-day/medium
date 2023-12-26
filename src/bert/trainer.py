import glob
import time
import torch
import natsort
import logging
import datetime
from pathlib import Path
from bert.bert import BERT
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from bert.dump_sentences import DumpStentences
from bert.scheduled_optim import ScheduledOptim
from torch.utils.data.distributed import DistributedSampler

from mtimer import MTimer
from bert.timer import Timer
from bert.dataset import BERTDatasetPrecached
from utils.config import Config

class BERTTrainer:
    def __init__(self,
                 config: Config,
                 model: BERT,
                 tokenizer,
                 ):
        self.config = config

        self.model = model
        self.tokenizer = tokenizer

        self.dump_sentences = DumpStentences(tokenizer)

        # TODO: add to args & config
        betas = (0.9, 0.999)
        weight_decay = 0.015

        self.criterion = torch.nn.NLLLoss(ignore_index=0).to(self.config.run.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.train.learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
        # self.optimizer_schedule = ScheduledOptim(self.optimizer, config.model.d_model, n_warmup_steps=10000)

        self._writer = None

        self.start_epoch = 0
        self.epoch = 0

    @property
    def writer(self):
        if self._writer is None:
            self._writer = SummaryWriter(str(self.config.run.logs_dir))
        return self._writer

    def before_epoch(self, epoch):
        pass

    def train(self):
        for self.epoch in range(self.config.train.start_epoch, self.config.train.end_epoch+1):
            loss = self.train_epoch(self.epoch)
            self.save_checkpoint(self.epoch + 1, -1, loss)

    def train_epoch(self, epoch):
        logging.info(f"Begin epoch {epoch}")

        loader, val_loader = self.before_epoch(epoch)

        losses = []
        self.train_timer = Timer()
        for i, data in enumerate(loader):
            sentence, labels = data
            sentence = sentence.to(self.config.run.device)
            labels = labels.to(self.config.run.device)

            mlm_out = self.model(sentence)

            eval_flag = (i + 1) % self.config.train.val_interval == 0
            if False and val_flag:
                # import numpy as np
                # np.set_printoptions(formatter={'float': '{:0.2f}'.format})
                # print(mlm_out.detach().cpu().numpy()[0,0,:])

                print("=" * 70 )
                predicted = self.dump_sentences.batched_debug(sentence, labels, mlm_out)
                print("\n".join(predicted[:5]))
                print("=" * 70 )

            loss = self.criterion(mlm_out.transpose(1, 2), labels)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            # self.optimizer_schedule.zero_grad()
            loss.backward()

            self.optimizer.step()
            # self.optimizer_schedule.step_and_update_lr()

            if eval_flag:
                if epoch % 2 == 1:
                    self.training_summary(losses, val_loader)
                else:
                    self.training_summary(losses, None) # , val_loader)

        self.training_summary(losses, val_loader)

        return loss

    def training_summary(self, losses, val_loader=None):
        # minimum number of batches before we start printing summary
        n = self.config.train.val_interval // 2

        # skip summary if
        if len(losses) < n:
            return

        # average over the last n batches
        loss = sum(losses[-n:]) / n
        if self.config.run.parallel_mode == 'ddp':
            loss = torch.tensor(loss).to(self.config.run.device)
            torch.distributed.all_reduce(loss)
            loss = loss.item() / torch.distributed.get_world_size()

        if val_loader is None:
            val_loss = None
        else:
            timer = Timer("val loss")
            val_loss = self.val_loss(val_loader)
            if self.config.run.is_primary:
                logging.info(timer.step())

            # if we are in DDP, we need to average the loss across all processes
            if self.config.run.parallel_mode == 'ddp':
                val_loss = torch.tensor(val_loss).to(self.config.run.device)
                torch.distributed.all_reduce(val_loss)
                val_loss = val_loss.item() / torch.distributed.get_world_size()

        passed = len(losses) / self.batch_count
        global_step = self.epoch * self.batch_count + len(losses)
        elapsed = self.train_timer.elapsed()
        items = [
            time.strftime('%H:%M:%S', time.gmtime(elapsed)),
            f"(r:{self.estimate_remaining_time(passed, elapsed)})",
            f"Epocn {self.epoch}",
            f"{len(losses)} / {self.batch_count} ({passed:6.2%})",
            f"loss: {loss:6.2f}",
        ]
        if val_loss is not None:
            items.append(f"Eval loss: {val_loss:6.2f}")

        if self.config.run.is_primary:
            self.writer.add_scalar("loss", loss, global_step=global_step)
            if val_loss is not None:
                self.writer.add_scalar("val_loss", val_loss, global_step=global_step)

        text = " | ".join(items)
        logging.info(text)
        # logging.info("\n".join(["-" * 70, text, "-" * 70]))


    def val_loss(self, loader):
        losses = []
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(loader):
                sentence, labels = data
                sentence = sentence.to(self.config.run.device)
                labels = labels.to(self.config.run.device)

                mlm_out = self.model(sentence)

                loss = self.criterion(mlm_out.transpose(1, 2), labels)

                losses.append(loss.item())
        self.model.train()
        loss = sum(losses) / len(losses)
        return loss


    @staticmethod
    def estimate_remaining_time(passed: float, elapsed: float):
        if passed <= 0:
            return "00:00:00"
        remaining = elapsed / passed - elapsed
        formatted = time.strftime('%H:%M:%S', time.gmtime(remaining))
        return formatted

    def save_checkpoint(self, epoch: int, index: int, loss: float):
        # skip checkpoint if this is not the main process
        if not self.config.run.is_primary:
            return

        is_wrapped = self.is_model_wrapped()

        global_step = epoch * self.batch_count + index
        start_time = time.time()
        name = f"bert_epoch{epoch}_index{global_step}_{datetime.datetime.utcnow().timestamp():.0f}.pt"
        torch.save({
            'version': 0.1,
            'epoch': epoch,
            'model_state_dict': (self.model.module if is_wrapped else self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, self.config.run.checkpoints_dir / name)

        # remove old checkpoints

        checkpoints = natsort.natsorted(self.config.run.checkpoints_dir.glob("*.pt"))
        if len(checkpoints) > self.config.train.max_checkpoints:
            for checkpoint in checkpoints[:-self.config.train.max_checkpoints]:
                checkpoint.unlink()

        text = "\n".join([
            "",
            "=" * 70,
            f"Model saved as '{name}' for {time.time() - start_time:.2f} secs",
            "=" * 70,
            ""
        ])
        logging.info(text)


    def load_checkpoint(self):
        path = Path(self.config.train.checkpoint)
        if not path.exists():
            raise ValueError(f"Checkpoint {path} does not exist.")
        logging.info(f"Restoring model {path}")

        # use map_location='cpu' if GPU memory an issue (broadcasting required in that case!)
        checkpoint = torch.load(path, map_location=self.config.run.device)

        version = checkpoint.get('version', 0)
        self.epoch = checkpoint['epoch'] + (1 if version == 0 else 0)
        self.config.train.start_epoch = self.epoch

        is_wrapped = self.is_model_wrapped()
        # in DDP, load state dict into the underlying model, otherwise, load it directly
        (self.model.module if is_wrapped else self.model).load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.config.run.parallel_mode == 'ddp':
            for param in self.model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        logging.info("=" * 70)
        logging.info("Model is restored.")
        logging.info("=" * 70)

    def is_model_wrapped(self):
        result = self.config.run.parallel_mode in ('dp', 'ddp')
        return result

# class BERTTrainerSingleDataset(BERTTrainer):

#     def __init__(self,
#                  model: BERT,
#                  logs_dir: Path,
#                  checkpoints_dir: Path,
#                  print_every: int,
#                  batch_size: int,
#                  learning_rate: float,
#                  epochs: int,
#                  tokenizer,
#                  device: str,
#                  train_dataset: Dataset,
#                  val_dataset: Dataset,
#                  d_model: int,
#                  ):
#             self.train_dataset = train_dataset
#             self.val_dataset = val_dataset
#             self.loader = None
#             self.val_loader = None
#             super().__init__(model, logs_dir, checkpoints_dir, print_every, batch_size, learning_rate, epochs, tokenizer, device, d_model)

#     def before_epoch(self, epoch):
#         if self.loader is None:
#             self.ds_size = len(self.dataset)
#             self.batch_size = self.batch_size
#             self.batch_count = self.ds_size // self.batch_size

#             # self.loader = DataLoader(dataset, batch_size, num_workers=1, shuffle=True, pin_memory=True)
#             self.loader = DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
#             self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=False, pin_memory=True)

#         return self.loader, self.val_loader


class BERTTrainerPreprocessedDatasets(BERTTrainer):
    def __init__(self,
                 config: Config,
                 model: BERT,
                tokenizer):
        super().__init__(config, model, tokenizer)


    def before_epoch(self, epoch):
        dataset = self.get_dataset(epoch, True)
        ds_size = len(dataset)
        batch_size = self.config.train.batch_size
        self.batch_count = ds_size // batch_size

        val_dataset = self.get_dataset(epoch, False)

        if self.config.run.parallel_mode == 'ddp':
            sampler = DistributedSampler(dataset)
            sampler.set_epoch(epoch)
            loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)

            val_sampler = DistributedSampler(val_dataset)
            val_sampler.set_epoch(epoch)
            val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return loader, val_loader

    def get_dataset(self, epoch, train):
        pattern = self.config.train.dataset_pattern if train else self.config.train.val_dataset_pattern

        pattern = str(self.config.run.datasets_dir / pattern)
        # add an optional .gz extension to the pattern
        dataset_files = glob.glob(pattern) + glob.glob(pattern + '.gz')
        if len(dataset_files) == 0:
            raise ValueError(f"Dataset files not found with pattern {pattern}")
        dataset_files = sorted(dataset_files)
        dataset_file = dataset_files[epoch % len(dataset_files)]

        logging.info(f"Epoch: {epoch} - Loading dataset from {dataset_file}")

        dataset = BERTDatasetPrecached(dataset_file)
        return dataset
