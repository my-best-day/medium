import time
import torch
import datetime
from pathlib import Path
from bert.bert import BERT
from bert.dataset import BERTDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class BERTTrainer2:
    def __init__(self,
                 model: BERT,
                 dataset: BERTDataset,
                 log_dir: Path,
                 checkpoint_dir: Path,
                 print_every: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 device: str = 'cpu',
                 tokenizer=None):
        self.model = model
        self.print_every = print_every
        self.epochs = epochs
        self.device = device
        self.tokenizer = tokenizer

        self.ds_size = len(dataset)
        self.batch_size = batch_size
        self.batch_count = self.ds_size // batch_size

        betas = (0.9, 0.999)
        weight_decay = 0.015

        self.loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
        self.criterion = torch.nn.NLLLoss(ignore_index=0).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                          lr=learning_rate, betas=betas) # , weight_decay=weight_decay)
        # self.optimizer_schedule = ScheduledOptim(self.optimizer, model.d_model, n_warmup_steps=10000)

        self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir

        self.start_epoch = 0
        self.epoch = 0

        # for p in self.model.parameters():
        #     print(p, p.nelement())

        total_parameters = sum([p.nelement() for p in self.model.parameters()])
        print(f"Total Parameters: {total_parameters:,}")


    def train(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            loss = self.train_epoch(self.epoch)
            self.save_checkpoint(self.epoch, -1, loss)

    def batched_debug(self, sentence, labels, mlm_out):
        with torch.no_grad():
            # sentence = sentence.clone()
            # labels = labels.clone()
            # mlm_out = mlm_out.clone()
            B, T, V = mlm_out.shape
            text = []
            for b in range(B):
                if any(element != 0 for element in labels[b]):
                    english = self.debug(sentence[b], labels[b], mlm_out[b])
                    text.append(english)
                    if len(text) >= 4:
                        break
            return text

    def debug(self, sentence, labels, mlm_out):
        english = []
        for i, id in enumerate(labels):
            if id != 0:
                predicted_id = mlm_out[i].argmax(axis=-1)
                token = f"/{self.convert_id_to_token(predicted_id)}/"
                print(f"{predicted_id} -> {token} : {describe(mlm_out[i])}")
            else:
                token = self.convert_id_to_token(sentence[i])
            english.append(token)
        english = self.convert_tokens_to_string(english)
        sentence2 = map(lambda i: labels[i] if sentence[i] == self.tokenizer.mask_token_id else sentence[i], range(len(sentence)))
        source = self.tokenizer.convert_ids_to_tokens(sentence2)
        source = self.convert_tokens_to_string(source)
        return f"{english}\n{source}"
    
    def convert_id_to_token(self, id):
        token = self.tokenizer.convert_ids_to_tokens([id])[0]
        return token
    
    def convert_tokens_to_string(self, tokens):
        import re
        text = self.tokenizer.convert_tokens_to_string(tokens)
        cleaned_text = re.sub(r'\[.*?\]\s*', '', text)
        return cleaned_text

    def train_epoch(self, epoch):
        print(f"Begin epoch {epoch + 1}")
        
        start_time = time.time()
        accumulated_loss = 0
        for i, data in enumerate(self.loader):
            sentence, labels = data['bert_input'], data['bert_label']

            sentence = sentence.to(self.device)
            labels = labels.to(self.device)

            mlm_out = self.model(sentence)
            if (i + 1) % self.print_every == 0:
                import numpy as np
                np.set_printoptions(formatter={'float': '{:0.2f}'.format})
                print(mlm_out.detach().cpu().numpy()[0,0,:])

                print("=" * 70 )
                predicted = self.batched_debug(sentence, labels, mlm_out)
                print("\n".join(predicted[:4]))
                print("=" * 70 )
            loss = self.criterion(mlm_out.transpose(1, 2), labels)
            accumulated_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.print_every == 0:
                elapsed = time.time() - start_time
                summary = self.training_summary(elapsed, (i+1), accumulated_loss)
                print(summary)

                accumulated_loss = 0

        return loss

    def training_summary(self, elsapsed, index, accumulated_loss):
        passed = index / self.batch_count
        global_step = self.epoch * self.batch_count + index
        mlm_loss = accumulated_loss / self.print_every
        result = " | ".join([
            time.strftime('%H:%M:%S', time.gmtime(elsapsed)),
            f"(r:{self.estimate_remaining_time(passed, elsapsed)})",
            f"Epocn {self.epoch + 1}",
            f"{index} / {self.batch_count} ({passed:6.2%})",
            f"MLM loss: {mlm_loss:6.2f}",
        ])
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
        name = f"bert_epoch{epoch}_index{index}_{datetime.datetime.utcnow().timestamp():.0f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, self.checkpoint_dir / name)

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
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = self.epoch
        print("Model is restored.")
        print("=" * 70)


def describe_tensor(tensor):
    """
    Returns descriptive statistics for a 1D PyTorch tensor.

    :param tensor: A 1D PyTorch tensor.
    :return: A dictionary with min, max, mean, standard deviation, 25th percentile, and 75th percentile.
    """
    if tensor.dim() != 1:
        raise ValueError("Tensor must be 1-dimensional.")

    desc_stats = {
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "stddev": tensor.std().item(),
        "25%": torch.quantile(tensor, 0.25).item(),
        "75%": torch.quantile(tensor, 0.75).item()
    }

    return desc_stats

def describe(tensor):
    """
    print the descriptive statistics for a 1D PyTorch tensor as one line of text.
    """
    stats = describe_tensor(tensor)
    print(" | ".join([f"{k}: {v:.2f}" for k, v in stats.items()]))