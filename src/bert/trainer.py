import time
import tqdm
import torch
from torch.optim import Adam

from bert.scheduled_optim import ScheduledOptim

class BERTTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, lr, weight_decay=0.01,
        betas=(0.9, 0.999), warmup_steps=10000, log_freq=10, device='cpu', eval_interval=200, eval_iters=50):

        self.device = device
        print(f"Device: {self.device} *** *** *** *** ***")
        self.model = model.to(device)
        self.train_data = train_dataloader
        self.eval_data = eval_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
            )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        total_parameters = sum([p.nelement() for p in self.model.parameters()])
        print(f"Total Parameters: {total_parameters:,}")
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    @torch.no_grad()
    def estimate_loss(self, data_loader):
        t0 = time.process_time()
        out = {}
        self.model.eval()
        # for split in ['train', 'val']:
        for split in ['train']:
            losses = torch.zeros(self.eval_iters)
            for k, batch in enumerate(self.eval_data):
                data = batch["bert_input"]
                label = batch["bert_label"]
                
                # move to device
                data = data.to(self.device)
                label = label.to(self.device)

                mask_lm_output = self.model(data)
                loss = self.criterion(mask_lm_output.transpose(1, 2), label)
                losses[k] = loss.item()
                if k == self.eval_iters - 1:
                    break
            out[split] = losses.mean()
        self.model.train()
        t1 = time.process_time()
        print(f"estimate_loss: {(t1 - t0):.3} secs")
        return out


    def iteration(self, epoch, data_loader, train=True):        
        avg_loss = 0.0

        # early_stopping = EarlyStopping(self.model, './', patience=7, verbose=True, delta=0.01)

        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        losses = {'train': 1e9}
        for i, data in data_iter:
            log_flag = i % self.eval_interval == 0 or i == len(data_iter) - 1
            if log_flag:
                losses = self.estimate_loss(data_loader)

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            mask_lm_output = self.model(data["bert_input"])

            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            # early_stopping(loss)
            # if early_stopping.stop_flag:
            #     print("Early Stopping *** *** *** *** *** *** ***")
            #     break

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
                "estimated": losses['train'].item()
            }

            # if i % self.log_freq == 0:
            if log_flag:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}"
        ) 
