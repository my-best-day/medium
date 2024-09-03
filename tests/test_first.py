import unittest
import tempfile
import shutil
from pathlib import Path
import torch
from torch_configurator import TorchConfigurator
from utils.config import Config, RunConfig, TrainConfig, ModelConfig
import abc


class _TestConfiguration(unittest.TestCase, abc.ABC):
    def setUp(self):
        self.setup_base_dir_tree()
        self.config = self.create_config(self.base_dir)
        self.task_handler = self.create_task_handler(self.config)
        self.configurator = TorchConfigurator(self.config, self.task_handler)
        self.configurator.configure()
        self.model = self.configurator.model
        self.trainer = self.configurator.trainer

    def tearDown(self):
        # Clean up after each test
        self.remove_base_dir()

    @abc.abstractmethod
    def create_config(self, base_dir):
        pass

    @abc.abstractmethod
    def create_task_handler(self, config):
        pass

    def assertHasAttr(self, obj, attr):
        self.assertTrue(hasattr(obj, attr), f"'{attr}' attribute not found in {obj}")

    def test_model_creation(self):
        model = self.configurator.model
        self.assertIsNotNone(model)
        self.assertIsInstance(model, torch.nn.Module)
        transformer_attribute_name = {
            'mlm': 'bert',
            'sst2': 'bert',
            'cola': 'bert',
            'gpt': 'gpt',
        }[self.config.model.task_type]
        self.assertHasAttr(model, transformer_attribute_name)
        self.assertHasAttr(model, 'lm_head')

    def test_optimizer_creation(self):
        optimizer = self.configurator.optimizer
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        # two param groups: when weight decay is applied, and when it is not.
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_model_forward_pass(self):
        import time
        start_time = time.time()

        model = self.model
        input_ids, labels = self.trainer.get_batch('train', True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Get Batch {elapsed_time:.4f} seconds")

        batch_size = self.config.train.batch_size
        seq_len = self.config.model.seq_len
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertEqual(input_ids.shape, (batch_size, seq_len))
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(labels.shape, (batch_size, seq_len))

        logits = model(input_ids)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Forward pass {elapsed_time:.4f} seconds")

        vocab_size = self.configurator.tokenizer.vocab_size
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    def test_trainer_attributes(self):
        self.trainer.get_batch('train', True)
        trainer = self.configurator.trainer

        self.assertIsNotNone(trainer)
        self.assertHasAttr(trainer, 'model')
        self.assertHasAttr(trainer, 'optimizer')
        self.assertEqual(trainer.iters, 0)
        self.assertEqual(trainer.start_iter, 0)

    def setup_base_dir_tree(self):
        self.base_dir = Path(tempfile.mkdtemp())
        runs_dir = self.base_dir / 'runs'
        runs_dir.mkdir(parents=True, exist_ok=True)

        self.setup_vocab_dir()
        self.setup_datasets_dir()

    @abc.abstractmethod
    def setup_vocab_dir(self):
        pass

    @abc.abstractmethod
    def setup_datasets_dir(self):
        pass

    def remove_base_dir(self):
        shutil.rmtree(self.base_dir)


class TestMlmConfiguration(_TestConfiguration):
    def create_config(self, base_dir):
        return create_mlm_config(base_dir)

    def setup_vocab_dir(self):  # NOSONAR
        """
        Set up the vocabulary directory for MLM configuration.
        This method is called by setup_base_dir_tree in the abstract superclass.
        """
        src_dir = Path('tests/resources/base_dir')

        vocab_dir = self.base_dir / 'vocab'
        vocab_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = vocab_dir / 'vocab.txt'
        shutil.copyfile(src_dir / 'vocab' / 'vocab.txt', vocab_path)

    def setup_datasets_dir(self):    # NOSONAR
        """
        Set up the datasets directory for MLM configuration.
        This method is called by setup_base_dir_tree in the abstract superclass.
        """
        src_dir = Path('tests/resources/base_dir')
        datasets_dir = self.base_dir / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)
        src_datasets_dir = src_dir / 'datasets'
        for filename in src_datasets_dir.iterdir():
            if 'mlm1_12_1' in filename.name:
                shutil.copy(filename, datasets_dir)

    def create_task_handler(self, config):  # NOSONAR
        from task.mlm.mlm_task_handler import MlmTaskHandler
        task_handler = MlmTaskHandler(self.config)
        return task_handler


class TestGptConfiguration(_TestConfiguration):
    def create_config(self, base_dir):
        return create_gpt_config(base_dir)

    def setup_vocab_dir(self):  # NOSONAR
        """
        MockGptTaskHandler does not need a vocab directory or files.
        This method is called by setup_base_dir_tree in the abstract superclass.
        """
        pass

    def setup_datasets_dir(self):  # NOSONAR
        """
        Set up the datasets directory for GPT configuration.
        This method is called by setup_base_dir_tree in the abstract superclass.
        """
        datasets_dir = self.base_dir / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)

        src_dir = Path('tests/resources/base_dir')
        src_datasets_dir = src_dir / 'datasets'
        for filename in src_datasets_dir.iterdir():
            if 'gpt_2' in filename.name:
                shutil.copy(filename, datasets_dir)

    def create_task_handler(self, config):
        from .task.gpt.mock_gpt_task_handler import MockGptTaskHandler
        task_handler = MockGptTaskHandler(self.config)
        return task_handler


def get_mlm_args(base_dir):
    model_args = {
        'task_type': 'mlm',
        'seq_len': 12,
        'd_model': 16,
        'n_layers': 1,
        'heads': 2
    }

    train_args = {
        "batch_size": 2,
        "val_interval": 2,
        "dataset_pattern": 'train_mlm1_12_*.msgpack',
        "val_dataset_pattern": 'val_mlm1_12_*.msgpack',
        "test_dataset_pattern": 'test_mlm1_12_*.msgpack',
        "weight_decay": 0.01,
        "dataset_percentage": 1.0,
        "val_dataset_percentage": 1.0,

        "learning_rate": 0.01,
        "min_learning_rate": 0.005,
        "warmup_iters": 4,
        "lr_decay_iters": 10,
        "max_iters": 12,

        "val_iters": 4,
        "test_iters": 4,

        "dropout": 0.1,
        "checkpoint": None,

        "switch_training": False,
        "test": False,

        "max_checkpoints": 1,
        "lr_scheduler": None,
    }

    run_args = {
        'base_dir': base_dir,
        'run_id': 1,
        'parallel_mode': 'single',
        'nproc': 1,
        'dist_master_addr': '127.0.0.1',
        'dist_master_port': '12233',
        'dist_backend': 'nccl',
        'wandb': False,
        'compile': True,
        'async_to_device': True,
        'fused_adamw': True,
        'flash': True,
        'datasets_dir': None,
        'run_dir': None,
        'logs_dir': None,
        'checkpoints_dir': None,
        'local_rank': None,
        'device': None,
        'is_primary': True,
        'case': 'dickens',
    }

    return model_args, train_args, run_args


def create_mlm_config(base_dir):
    model_args, train_args, run_args = get_mlm_args(base_dir)

    model_config = ModelConfig(**model_args)
    train_config = TrainConfig(**train_args)
    run_config = RunConfig(**run_args)

    config = Config(
        model=model_config,
        train=train_config,
        run=run_config
    )
    return config


def create_gpt_args(base_dir):
    model_args, train_args, run_args = get_mlm_args(base_dir)

    model_args['task_type'] = 'gpt'
    model_args['seq_len'] = 14
    model_args['n_layers'] = 2
    model_args['heads'] = 2

    train_args['batch_size'] = 2
    train_args['val_interval'] = 3
    train_args['val_iters'] = 1
    train_args['test_iters'] = 1

    train_args['dataset_pattern'] = 'train_mock_gpt_2.msgpack'
    train_args['val_dataset_pattern'] = 'val_mock_gpt_2.msgpack'
    train_args['test_dataset_pattern'] = 'test_mock_gpt_2.msgpack'

    return model_args, train_args, run_args


def create_gpt_config(base_dir):
    model_args, train_args, run_args = create_gpt_args(base_dir)

    model_config = ModelConfig(**model_args)
    train_config = TrainConfig(**train_args)
    run_config = RunConfig(**run_args)

    config = Config(
        model=model_config,
        train=train_config,
        run=run_config
    )

    return config


if __name__ == '__main__':
    unittest.main()
