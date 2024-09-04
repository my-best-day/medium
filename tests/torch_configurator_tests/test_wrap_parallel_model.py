import torch
import unittest
from unittest.mock import patch, MagicMock
from torch_configurator import TorchConfigurator


class TestWrapParallelModel(unittest.TestCase):

    def setUp(self):
        # Example configuration structure for testing
        self.config = MagicMock()
        self.config.run = MagicMock()
        self.task_handler = MagicMock()

    def test_wrap_parallel_model_single(self):
        self.config.run.parallel_mode = 'single'

        mock_model = MagicMock()

        obj = TorchConfigurator(self.config, self.task_handler)
        wrapped_model = obj.wrap_parallel_model(mock_model)

        self.assertEqual(wrapped_model, mock_model)

    def test_wrap_parallel_model_dp(self):
        self.config.run.parallel_mode = 'dp'
        mock_model = MagicMock()

        obj = TorchConfigurator(self.config, self.task_handler)
        wrapped_model = obj.wrap_parallel_model(mock_model)

        self.assertIsInstance(wrapped_model, torch.nn.DataParallel)
        self.assertEqual(wrapped_model.module, mock_model)

    @patch('torch.nn.parallel.DistributedDataParallel')
    def test_wrap_parallel_model_ddp(self, mock_ddp):
        self.config.run.dist_backend = 'nccl'
        self.config.run.parallel_mode = 'ddp'
        self.config.run.local_rank = 0

        mock_model = MagicMock()

        obj = TorchConfigurator(self.config, self.task_handler)
        wrapped_model = obj.wrap_parallel_model(mock_model)

        mock_ddp.assert_called_once_with(mock_model, device_ids=[0])
        self.assertEqual(wrapped_model, mock_ddp.return_value)

    def test_wrap_parallel_model_unknown_mode(self):
        self.config.run.parallel_mode = 'unknown'

        mock_model = MagicMock()

        obj = TorchConfigurator(self.config, self.task_handler)

        with self.assertRaises(ValueError):
            obj.wrap_parallel_model(mock_model)


if __name__ == '__main__':
    unittest.main()
