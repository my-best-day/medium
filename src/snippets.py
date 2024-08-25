    def get_lm_model(self):
        """
        Return a language model for the given config and tokenizer.
        A transformer with a language model head.
        """
        from transformer.lm.mlm.bert_mlm_model import BertMlmModel
        from transformer.lm.classifier.bert_classifier_model import BertClassifierModel
        from transformer.lm.gpt.gpt_model import GptModel

        vocab_size = len(self.tokenizer.vocab)

        transformer_model = self.get_transformer_model()

        task_type = self.config.model.task_type
        if task_type in ('cola', 'sst2'):
            lm_model = BertClassifierModel(transformer_model, 2)
        elif task_type == 'mlm':
            lm_model = BertMlmModel(transformer_model, vocab_size, apply_softmax=False)
        elif task_type == 'gpt':
            lm_model = GptModel(transformer_model, vocab_size)
        else:
            raise ValueError('Unknown task type %s', task_type)

        return lm_model



    def resume_from_checkpoint(self):
        config = self.config

        path = config.train.checkpoint
        if path is None:
            return

        logging.info("Resuming from checkpoint at %s", path)
        # use map_location='cpu' if GPU memory an issue (broadcasting required in that case!)
        checkpoint = torch.load(path, map_location=config.run.device)

        # load model state
        model_state = checkpoint['model']

        if config.model.task_type != 'mlm':
            # Filter out 'mask_lm' parameters
            model_state = {k: v for k, v in checkpoint['model'].items() if 'mask_lm' not in k}

        is_wrapped = self.is_model_wrapped()
        # in DDP, load state dict into the underlying model, otherwise, load it directly
        (self.model.module if is_wrapped else self.model).load_state_dict(model_state, strict=False)

        if config.run.parallel_mode == 'ddp':
            for param in self.model.module.parameters():
                torch.distributed.broadcast(param.data, src=0)

        if config.model.task_type == 'mlm':
            # load optimizer state
            optimizer_state = checkpoint['optimizer']
            self.optimizer.load_state_dict(optimizer_state)

            # load trainer state
            iteration = checkpoint['iter']
            val_loss = checkpoint['val_loss']
        else:
            iteration = 0
            val_loss = 0

        self.trainer.start_iter = iteration
        self.trainer.iter = self.trainer.start_iter

        self.trainer.best_val_loss = val_loss

        logging.info("Resuming from iteration %s, with val loss %s", iteration, val_loss)


        # if self.config.model.task_type in ['mlm', 'gpt']:
        #     # For MLM or GPT tasks
        #     loss_logits = logits.transpose(1, 2)  # Shape: [batch_size, vocab_size, seq_len]
        #     loss = torch.nn.functional.cross_entropy(loss_logits, Y, ignore_index=0)
        # else:
        #     # For other tasks
        #     loss_logits = logits  # Shape: [batch_size, num_classes]
        #     loss = torch.nn.functional.cross_entropy(loss_logits, Y)  # Y should be [batch_size]



        {
            'format': f'{self.config.model.task_type}.1',
            'version': 1.0,
            'iter': iter,
            'model': (self.model.module if is_wrapped else self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict(),
        }


        # flatten tensors for comparison
        y_flat = Y.view(-1)
        predicted_flat = predicted.view(-1)


        if self.config.model.task_type == 'gpt':
            correct = SKIP_ACCURACY
            total = SKIP_ACCURACY
        elif self.config.model.task_type == 'mlm':
            # mask: ignore padding (assumed 0) and focus on masked tokens (assumed non zero)
            mask = (y_flat != 0)
            total += mask.sum().item()
            correct += (predicted_flat[mask] == y_flat[mask]).sum().item()
        else:
            total += Y.size(0)
            correct += (predicted == Y).sum().item()


        if self.config.model.task_type == 'gpt':
            dataset = self.get_gpt_dataset(epoch, split)
        elif self.config.model.task_type == 'mlm':
            dataset = self.get_mlm_dataset(epoch, split)
        elif self.config.model.task_type == 'cola':
            dataset = self.get_cola_dataset(split)
        elif self.config.model.task_type == 'sst2':
            dataset = self.get_sst2_dataset(split)
        else:
            raise ValueError(f"Unknown dataset: {self.config.run.dataset}")
        return dataset


    def get_gpt_dataset(self, epoch, split):
        from data.gpt.gpt_token_ids_dataset import GptTokenIdsDataset
        seq_len = self.config.model.seq_len
        percentage = self.get_percentage(split)
        dataset_file = self.find_dataset_file(epoch, split)
        dataset = GptTokenIdsDataset(dataset_file, seq_len, percentage)
        return dataset


    def get_percentage(self, split):
        if split == 'train':
            percentage = self.config.train.dataset_percentage
        elif split == 'val':
            percentage = self.config.train.val_dataset_percentage
        else:
            raise ValueError(f"Unknown split: {split}")
        return percentage

    def get_cola_dataset(self, split):
        assert split in ('train', 'val')
        from data.cola_dataset import ColaDataset
        if split == 'train':
            filename = 'in_domain_train.tsv'
        elif split == 'val':
            filename = 'in_domain_dev.tsv'
        path = self.config.run.datasets_dir / filename
        dataset = ColaDataset(path, self.tokenizer, self.config.model.seq_len)
        return dataset

    def get_sst2_dataset(self, split):
        assert split in ('train', 'val')
        from data.sst2_dataset import Sst2Dataset
        if split == 'train':
            prefix = 'train'
        elif split == 'val':
            prefix = 'validation'
        filename = f'{prefix}-00000-of-00001.parquet'
        path = self.config.run.datasets_dir / filename
        dataset = Sst2Dataset(path, self.tokenizer, self.config.model.seq_len)
        return dataset
