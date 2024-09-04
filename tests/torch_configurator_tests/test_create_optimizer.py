import torch
import unittest
from unittest.mock import patch, MagicMock
from torch_configurator import TorchConfigurator


class TestCreateOptimizer(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.run = MagicMock()
        self.config.train = MagicMock()
        self.config.run.is_primary = True
        self.config.train.weight_decay = 0.01
        self.config.train.learning_rate = 0.001
        self.task_handler = MagicMock()
        self.model = MagicMock()
        self.param1 = torch.nn.Parameter(torch.randn(10))
        self.param2 = torch.nn.Parameter(torch.randn(10, 10))
        self.model.parameters.return_value = [self.param1, self.param2]
        self.configurator = TorchConfigurator(self.config, self.task_handler)
        self.configurator.model = self.model

    @patch('torch.optim.AdamW')
    def test_create_optimizer(self, mock_adamw):
        optimizer = self.configurator.create_optimizer()
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, MagicMock)  # Since mock_adamw is a MagicMock
        mock_adamw.assert_called_once_with(
            [{'params': [self.param2], 'weight_decay': 0.01},
             {'params': [self.param1], 'weight_decay': 0.0}],
            lr=0.001, betas=(0.9, 0.999)
        )

    @patch('torch.optim.AdamW')
    @patch.object(TorchConfigurator, 'should_use_fused_adamw', return_value=True)
    def test_create_optimizer_with_fused(self, mock_should_use_fused_adamw, mock_adamw):
        optimizer = self.configurator.create_optimizer()
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, MagicMock)  # Since mock_adamw is a MagicMock
        mock_adamw.assert_called_once_with(
            [{'params': [self.param2], 'weight_decay': 0.01},
                {'params': [self.param1], 'weight_decay': 0.0}],
            lr=0.001, betas=(0.9, 0.999), fused=True
        )

    def test_optimizer_param_groups(self):
        optimizer = self.configurator.create_optimizer()
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01)
        self.assertEqual(optimizer.param_groups[0]['params'], [self.param2])
        self.assertEqual(optimizer.param_groups[1]['weight_decay'], 0.0)
        self.assertEqual(optimizer.param_groups[1]['params'], [self.param1])


if __name__ == '__main__':
    unittest.main()
