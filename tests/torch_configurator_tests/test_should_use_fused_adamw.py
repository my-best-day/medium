import torch
import unittest
from unittest.mock import MagicMock, patch
from torch_configurator import TorchConfigurator


class TestShouldUseFusedAdamW(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.task_handler = MagicMock()
        self.configurator = TorchConfigurator(self.config, self.task_handler)

    @patch('inspect.signature')
    def test_should_use_fused_adamw(self, mock_inspect_signature):
        # config.run.fused_adamw, fused_available, device_type, expected_result
        states = [
            [False, True, 'cuda', False],
            [False, True, 'cpu', False],
            [False, False, 'cuda', False],
            [False, False, 'cpu', False],
            [True, True, 'cuda', True],
            [True, True, torch.device('cuda'), True],
            [True, True, 'cpu', False],
            [True, True, torch.device('cpu'), False],
            [True, False, 'cuda', False],
            [True, False, torch.device('cuda'), False],
            [True, False, 'cpu', False],
            [True, False, torch.device('cpu'), False],
        ]

        for config_flag, fused_available, device, expected in states:
            with self.subTest(config_flag=config_flag, fused_available=fused_available,
                              device=device, expected=expected):
                self.config.run.fused_adamw = config_flag
                mock_inspect_signature.return_value = MagicMock()
                mock_inspect_signature.return_value.parameters = ['fused'] if fused_available else []
                self.config.run.device = device
                result = self.configurator.should_use_fused_adamw()
                self.assertEqual(result, expected)
