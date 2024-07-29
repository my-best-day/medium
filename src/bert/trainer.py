import glob
import time
import torch
import natsort
import logging
import datetime
from pathlib import Path
from bert.bert import BERT
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from bert.dump_sentences import DumpStentences
from torch.utils.data.distributed import DistributedSampler

from bert.timer import Timer
from bert.dataset import BERTDatasetPrecached
from utils.config import Config


class BERTTrainer:
    def __init__(self,
                 config: Config,
                 model: BERT,
                 optimizer,
                 tokenizer,
                 ):
        self.config = config

        self.model = model
        self.tokenizer = tokenizer

        self.dump_sentences = DumpStentences(tokenizer)

        # TODO: add to args & config
        betas = (0.9, 0.999)
        weight_decay = self.config.train.weight_decay

        self.criterion = torch.nn.NLLLoss(ignore_index=0).to(self.config.run.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.train.learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
        self.lr_sched = get_lr_scheduler(self.optimizer, self.config.train.lr_scheduler)

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
        timer = Timer("epoch time")
        for self.epoch in range(self.config.train.start_epoch, self.config.train.end_epoch):
            if Path('./stop').exists() or Path('./stop_now').exists():
                logging.info("Stopping training because file './stop' or './stop_now' exists.")
                break
            loss = self.train_epoch(self.epoch)
            self.lr_sched.step()
            self.save_checkpoint(self.epoch + 1, -1, loss)
            logging.info(timer.step(f"Epoch {self.epoch}", restart=True))

    def train_epoch(self, epoch):
        current_lr = self.optimizer.param_groups[0]['lr']
        logging.info(f"Begin epoch {epoch} with learning rate {current_lr}")

        loader, val_loader = self.before_epoch(epoch)

        losses = []
        self.train_timer = Timer()
        for i, data in enumerate(loader):
            sentence, labels = data
            sentence = sentence.to(self.config.run.device)
            labels = labels.to(self.config.run.device)

            mlm_out = self.model(sentence)

            eval_flag = (i + 1) % self.config.train.val_interval == 0
            val_flag = False  # (i + 1) % (self.config.train.val_interval * 2) == 0
            if False and val_flag:
                # import numpy as np
                # np.set_printoptions(formatter={'float': '{:0.2f}'.format})
                # print(mlm_out.detach().cpu().numpy()[0,0,:])

                print("=" * 70)
                predicted = self.dump_sentences.batched_debug(sentence, labels, mlm_out)
                print("\n".join(predicted[:5]))
                print("=" * 70)

            loss = self.criterion(mlm_out.transpose(1, 2), labels)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if eval_flag:
                if val_flag:
                    self.training_summary(losses, val_loader)
                else:
                    self.training_summary(losses, None)  # , val_loader)

            if Path('./stop_now').exists():
                logging.info("Stopping in the middle of the epoch training because file "
                             "'./stop_now' exists.")
                break

        self.training_summary(losses, val_loader)

        return loss

    def training_summary(self, losses, val_loader=None):
        # minimum number of batches before we start printing summary
        n = 4  # self.config.train.val_interval // 2

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
            val_accuracy = None
        else:
            timer = Timer("val loss")
            val_loss, val_accuracy = self.val_loss(val_loader)
            if self.config.run.is_primary:
                logging.info(timer.step())

            # if we are in DDP, we need to average the loss across all processes
            if self.config.run.parallel_mode == 'ddp':
                val_loss = torch.tensor(val_loss).to(self.config.run.device)
                torch.distributed.all_reduce(val_loss)
                val_loss = val_loss.item() / torch.distributed.get_world_size()

                val_accuracy = torch.tensor(val_accuracy).to(self.config.run.device)
                torch.distributed.all_reduce(val_accuracy)
                val_accuracy = val_accuracy.item() / torch.distributed.get_world_size()

        n_losses = len(losses)
        if self.config.run.parallel_mode == 'ddp':
            global_step = self.epoch * self.batch_count + \
                n_losses * torch.distributed.get_world_size()
            n_losses *= torch.distributed.get_world_size()
        else:
            global_step = self.epoch * self.batch_count + n_losses
        # global_step *= self.config.train.batch_size

        n_losses = min(n_losses, self.batch_count)
        passed = n_losses / self.batch_count

        elapsed = self.train_timer.elapsed()
        items = [
            time.strftime('%H:%M:%S', time.gmtime(elapsed)),
            f"(r:{self.estimate_remaining_time(passed, elapsed)})",
            f"Epocn {self.epoch}",
            f"{n_losses} / {self.batch_count} ({passed:6.2%})",
            f"loss: {loss:6.2f}",
        ]
        if val_loss is not None:
            items.append(f"Eval loss: {val_loss:6.2f}")
        if val_accuracy is not None:
            items.append(f"Eval accuracy: {val_accuracy:6.2%}")

        self._log_progress(global_step, loss, val_loss, val_accuracy)

        text = " | ".join(items)
        logging.info(text)

    def _log_progress(self, global_step, loss, val_loss, val_accuracy):
        if not self.config.run.is_primary:
            return

        self.writer.add_scalar("train_loss", loss, global_step=global_step)
        if self.config.run.wandb:
            import wandb
            wandb.log({"train_loss": loss}, step=global_step)
        if val_loss is not None:
            self.writer.add_scalar("val_loss", val_loss, global_step=global_step)
            if self.config.run.wandb:
                wandb.log({"val_loss": val_loss}, step=global_step)
        if val_accuracy is not None:
            self.writer.add_scalar("val_accuracy", val_accuracy, global_step=global_step)
            if self.config.run.wandb:
                wandb.log({"val_accuracy": val_accuracy}, step=global_step)

    def val_loss(self, loader):
        losses = []
        total = 0
        correct = 0
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(loader):
                sentence, labels = data
                sentence = sentence.to(self.config.run.device)
                labels = labels.to(self.config.run.device)

                mlm_out = self.model(sentence)

                loss = self.criterion(mlm_out.transpose(1, 2), labels)
                losses.append(loss.item())

                # calculate accuracy
                _, predicted = torch.max(mlm_out.data, 2)
                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()

                if i == 0 and self.config.run.case == 'movies' and (self.epoch + 1) % 10 == 0:
                    predicted = self.dump_sentences.batched_debug(sentence, labels, mlm_out)
                    logging.info("\n".join(predicted[:30]))

        self.model.train()
        loss = sum(losses) / len(losses)
        accuracy = correct / total
        return loss, accuracy

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
        timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
        name = f"bert_epoch{epoch}_index{global_step}_{timestamp:.0f}.pt"
        checkpoint_path = self.config.run.checkpoints_dir / name

        # initial version save epoch - 1
        # 0.1 fixed the epoch
        # 0.2 added lr_sched state
        torch.save({
            'version': 0.2,
            'epoch': epoch,
            'model_state_dict': (self.model.module if is_wrapped else self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_sched.state_dict(),
            'loss': loss
        }, checkpoint_path)

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
        model_state = checkpoint['model_state_dict']
        (self.model.module if is_wrapped else self.model).load_state_dict(model_state)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if version >= 0.2:
            self.lr_sched.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.config.run.parallel_mode == 'ddp':
            for param in self.model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        logging.info("=" * 70)
        logging.info("Model is restored.")
        logging.info("=" * 70)

    def is_model_wrapped(self):
        result = self.config.run.parallel_mode in ('dp', 'ddp')
        return result


class LRSchedulerNoop:
    def step(self):
        pass

    def state_dict(self):
        return {}


def get_lr_scheduler(optimizer, lr_scheduler_arg):
    if lr_scheduler_arg is None:
        return LRSchedulerNoop()
    if lr_scheduler_arg.startswith('steplr'):
        _, step_size, gamma = lr_scheduler_arg.split(':')
        step_size = int(step_size)
        gamma = float(gamma)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise Exception(f'Unknown learning rate scheduler {lr_scheduler_arg}. '
                        'Valid values are warmup, cosine.')


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
#             super().__init__(model, logs_dir, checkpoints_dir, print_every, batch_size,
#                              learning_rate, epochs, tokenizer, device, d_model)

#     def before_epoch(self, epoch):
#         if self.loader is None:
#             self.ds_size = len(self.dataset)
#             self.batch_size = self.batch_size
#             self.batch_count = self.ds_size // self.batch_size

#             # self.loader = DataLoader(dataset, batch_size, num_workers=1, shuffle=True,
#                                        pin_memory=True)
#             self.loader = DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
#             self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=False,
#                                          pin_memory=True)

#         return self.loader, self.val_loader


class BERTTrainerPreprocessedDatasets(BERTTrainer):
    def __init__(self,
                 config: Config,
                 model: BERT,
                 optimizer,
                 tokenizer):
        super().__init__(config, model, optimizer, tokenizer)

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
            val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size,
                                    shuffle=False, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    pin_memory=True)

        return loader, val_loader

    def get_dataset(self, epoch, train):
        pattern = self.config.train.dataset_pattern if train else \
            self.config.train.val_dataset_pattern

        pattern = str(self.config.run.datasets_dir / pattern)
        # add an optional .gz extension to the pattern
        dataset_files = glob.glob(pattern) + glob.glob(pattern + '.gz')
        if len(dataset_files) == 0:
            raise ValueError(f"Dataset files not found with pattern {pattern}")
        dataset_files = sorted(dataset_files)
        dataset_file = dataset_files[epoch % len(dataset_files)]

        logging.info(f"Epoch: {epoch} - Loading dataset from {dataset_file}")

        percentage = self.config.train.dataset_percentage if train else \
            self.config.train.val_dataset_percentage
        dataset = BERTDatasetPrecached(dataset_file, percentage)
        return dataset
