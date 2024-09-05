import torch
import unittest
from unittest.mock import MagicMock
from torch_configurator import TorchConfigurator


class TestGetDecayNoDecayParameters(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.task_handler = MagicMock()
        self.configurator = TorchConfigurator(self.config, self.task_handler)

    def test_get_decay_no_decay_parameters(self):
        # Create parameters with various shapes and requires_grad attributes
        # 1D, requires_grad=True
        param1 = torch.nn.Parameter(torch.randn(10), requires_grad=True)
        # 2D, first dim is 1
        param2 = torch.nn.Parameter(torch.randn(1, 10), requires_grad=True)
        # 2D, requires_grad=True
        param3 = torch.nn.Parameter(torch.randn(10, 10), requires_grad=True)
        # 1D, requires_grad=False
        param4 = torch.nn.Parameter(torch.randn(10), requires_grad=False)
        # 2D, first dim is 1, requires_grad=False
        param5 = torch.nn.Parameter(torch.randn(1, 10), requires_grad=False)

        # Mock the model's parameters method to return these parameters
        self.configurator.model = MagicMock()
        self.configurator.model.parameters.return_value = [param1, param2, param3, param4, param5]

        # Call the method under test
        decay_params, no_decay_params = self.configurator.get_decay_no_decay_parameters()

        # Verify the results
        self.assertEqual(decay_params, [param3])
        self.assertEqual(no_decay_params, [param1, param2])


if __name__ == '__main__':
    unittest.main()
