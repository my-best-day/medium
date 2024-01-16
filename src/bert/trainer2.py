import glob
import time
import torch
import natsort
import inspect
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

from contextlib import nullcontext

from mtimer import MTimer
from bert.timer import Timer
from bert.dataset import BERTDatasetPrecached
from utils.config import Config

class BERTTrainer2:
    def __init__(self,
                 config: Config,
                 model: torch.nn.Module,
                 tokenizer,
                 ):
        self.config = config

        self.model = model
        self.tokenizer = tokenizer

        self.dump_sentences = DumpStentences(tokenizer)

        self.optimizer = self.configure_optimizer()

        self._writer = None

        self.iter_num = 0

        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

        # torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = self.config.run.device.type # 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    def configure_optimizer(self, model):
        # figure which parameters require weight decay
        seen = set()
        decay_params = []
        no_decay_params = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param in seen:
                continue
            seen.add(param)
            if len(param.shape) == 1 or param.shape[0] == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimization_groups = [
            {'params': decay_params, 'weight_decay': self.config.train.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        if self.config.run.is_primary:
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in no_decay_params)
            logging.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            logging.info(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_nodecay_params:,} parameters")

        fused_available_ = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use fused if available and or device type is 'cuda'
        use_fused = fused_available_ and self.config.run.device.type == 'cuda'
        if self.config.run.is_primary:
            logging.info(f"Using fused AdamW: {use_fused}")

        extra_args = dict()
        if use_fused:
            extra_args['fused'] = True

        optimizer = torch.optim.AdamW(
            optimization_groups,
            lr=self.config.train.learning_rate,
            betas=(0.9, 0.999),
            **extra_args
        )

        return optimizer

    @property
    def writer(self):
        if self._writer is None:
            self._writer = SummaryWriter(str(self.config.run.logs_dir))
        return self._writer

    def before_epoch(self, epoch):
        pass

    def train_loop(self, optimizer):
        iter = 0
        val_interval = self.config.train.val_interval
        eval_only = False
        while True:

            lr = self.get_lr(iter)
            for param_group in optimizer.paragam_groups:
                param_group['lr'] = lr

            if iter % val_interval == 0:
                losses = self.estimate_loss()
                # log losses - either her or inside estimate loss
                # how to handle train loss? I want to use the loss computed during training

            # TODO: what is eval-only mode?
            if eval_only:
                break

            with self.ctx:



    def train(self):
        timer = Timer("epoch time")
        for self.epoch in range(self.config.train.start_epoch, self.config.train.end_epoch):
            if Path('./stop').exists():
                logging.info("Stopping training because file './stop' exists.")
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
            val_flag = False # (i + 1) % (self.config.train.val_interval * 2) == 0
            if False and val_flag:
                # import numpy as np
                # np.set_printoptions(formatter={'float': '{:0.2f}'.format})
                # print(mlm_out.detach().cpu().numpy()[0,0,:])

                print("=" * 70 )
                predicted = self.dump_sentences.batched_debug(sentence, labels, mlm_out)
                print("\n".join(predicted[:5]))
                print("=" * 70 )

            # loss = self.criterion(mlm_out.transpose(1, 2), labels)
            loss = torch.nn.functional.cross_entropy(mlm_out.transpose(1, 2), labels, ignore_index=0)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if eval_flag:
                if val_flag:
                    self.training_summary(losses, val_loader)
                else:
                    self.training_summary(losses, None) # , val_loader)

            if Path('./stop_now').exists():
                logging.info("Stopping in the middle of the epoch training because file './stop_now' exists.")
                break

        self.training_summary(losses, val_loader)

        return loss

    def get_lr(self, it):
        import math

        lr = self.config.train.learning_rate
        min_lr = self.config.train.min_learning_rate

        warmup_iter = self.config.train.warmup_iter
        lr_decay_iters = self.config.train.lr_decay_iters

        if it < warmup_iter:
            lr = it * lr / warmup_iter

        elif it <= lr_decay_iters:
            ratio = (it - warmup_iter) / (lr_decay_iters - warmup_iter)
            assert 0 <= ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio)) # range 0..1
            lr = min_lr + coeff * (lr - min_lr)

        else: # it > lr_decay_iters:
            lr = min_lr

        return lr

    def training_summary(self, losses, val_loader=None):
        # minimum number of batches before we start printing summary
        n = 4 # self.config.train.val_interval // 2

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
            global_step = self.epoch * self.batch_count + n_losses * torch.distributed.get_world_size()
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

        self.writer.add_scalar("loss", loss, global_step=global_step)
        if self.config.run.wandb:
            import wandb
            wandb.log({"loss": loss}, step=global_step)
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
        name = f"bert_epoch{epoch}_index{global_step}_{datetime.datetime.utcnow().timestamp():.0f}.pt"
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
        (self.model.module if is_wrapped else self.model).load_state_dict(checkpoint['model_state_dict'])

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

def get_lr_scheduler(optimizer, lr_scheduler_arg):
    if lr_scheduler_arg is None:
        return LRSchedulerNoop()
    if lr_scheduler_arg.startswith('steplr'):
        _, step_size, gamma = lr_scheduler_arg.split(':')
        step_size = int(step_size)
        gamma = float(gamma)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise Exception(f'Unknown learning rate scheduler {lr_scheduler_arg}. Valid values are warmup, cosine.')



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

        percentage = self.config.train.dataset_percentage if train else self.config.train.val_dataset_percentage
        dataset = BERTDatasetPrecached(dataset_file, percentage)
        return dataset
