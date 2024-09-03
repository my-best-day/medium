import torch
import unittest
from unittest.mock import patch, MagicMock
from torch_configurator import TorchConfigurator


class TestInitModeSetDevice(unittest.TestCase):

    def setUp(self):
        # Example configuration structure for testing
        self.config = MagicMock()
        self.config.run = MagicMock()
        self.config.run.local_rank = 0  # example rank for DDP
        self.task_handler = MagicMock()

    @patch('torch.cuda.is_available')
    def test_single_mode_cuda_available(self, mock_is_available):
        mock_is_available.return_value = True

        # Call the method
        self.config.run.parallel_mode = 'single'
        obj = TorchConfigurator(self.config, self.task_handler)
        obj.init_mode_set_device()

        # Assert that the device was set to 'cuda'
        self.assertEqual(self.config.run.device, 'cuda')

    @patch('torch.cuda.is_available')
    def test_single_mode_cuda_not_available(self, mock_is_available):
        mock_is_available.return_value = False

        # Call the method
        self.config.run.parallel_mode = 'single'
        obj = TorchConfigurator(self.config, self.task_handler)
        obj.init_mode_set_device()

        # Assert that the device was set to 'cpu'
        self.assertEqual(self.config.run.device, 'cpu')

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_dp_mode_multiple_gpus(self, mock_device_count, mock_is_available):
        mock_is_available.return_value = True
        mock_device_count.return_value = 2

        # Call the method
        self.config.run.parallel_mode = 'dp'
        obj = TorchConfigurator(self.config, self.task_handler)
        obj.init_mode_set_device()

        # Assert that the device was set to 'cuda'
        self.assertEqual(self.config.run.device, 'cuda')

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_dp_mode_single_gpu(self, mock_device_count, mock_is_available):
        mock_is_available.return_value = True
        mock_device_count.return_value = 1

        # Call the method
        self.config.run.parallel_mode = 'dp'
        obj = TorchConfigurator(self.config, self.task_handler)

        with self.assertRaises(RuntimeError):
            obj.init_mode_set_device()

    @patch('torch.cuda.is_available')
    def test_ddp_mode_no_cuda(self, mock_is_available):
        mock_is_available.return_value = False

        # Call the method
        self.config.run.parallel_mode = 'ddp'
        obj = TorchConfigurator(self.config, self.task_handler)

        with self.assertRaises(RuntimeError):
            obj.init_mode_set_device()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.current_device')
    def test_ddp_mode_with_cuda(self, mock_current_device, mock_set_device, mock_is_available):
        mock_is_available.return_value = True
        mock_current_device.return_value = 0

        # Call the method
        self.config.run.parallel_mode = 'ddp'
        obj = TorchConfigurator(self.config, self.task_handler)
        obj.init_mode_set_device()

        # Assert that the correct device is set
        mock_set_device.assert_called_once_with(0)
        self.assertEqual(self.config.run.device, torch.device('cuda', 0))

    def test_unknown_mode(self):
        # Call the method with an unknown mode
        self.config.run.parallel_mode = 'unknown'
        obj = TorchConfigurator(self.config, self.task_handler)

        with self.assertRaises(ValueError):
            obj.init_mode_set_device()

    ############################################################
    from task.mlm.mlm_task_handler import MlmTaskHandler

    @patch.object(TorchConfigurator, 'wrap_parallel_model')
    @patch('torch.compile', return_value=MagicMock())
    def test_create_model_no_compile(self, mock_compile,
                                     mock_wrap_parallel_model):
        # Setup
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_task_handler = MagicMock()
        mock_task_handler.create_lm_model.return_value = mock_model
        mock_wrap_parallel_model.return_value = mock_model

        config = MagicMock()
        config.run.compile = False
        config.run.device = 'cuda'

        # Execute
        obj = TorchConfigurator(config, mock_task_handler)

        model = obj.create_model()

        # Assert
        mock_task_handler.create_lm_model.assert_called_once()
        mock_compile.assert_not_called()
        mock_wrap_parallel_model.assert_called_once_with(mock_model)
        self.assertEqual(model, mock_wrap_parallel_model.return_value)
        mock_model.to.assert_called_once_with('cuda')

    @patch.object(TorchConfigurator, 'wrap_parallel_model')
    def test_create_model_with_compile(self, mock_wrap_parallel_model):
        # Setup
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_task_handler = MagicMock()
        mock_task_handler.create_lm_model.return_value = mock_model
        mock_wrap_parallel_model.return_value = mock_model

        config = MagicMock()
        config.run.compile = True
        config.run.device = 'cuda'

        # Execute
        obj = TorchConfigurator(config, mock_task_handler)

        with patch('torch.compile', return_value=mock_model) as mock_compile:
            model = obj.create_model()

            # Assert
            mock_task_handler.create_lm_model.assert_called_once()
            mock_compile.assert_called_once_with(mock_model)
            mock_wrap_parallel_model.assert_called_once_with(mock_compile.return_value)
            self.assertEqual(model, mock_wrap_parallel_model.return_value)
            mock_model.to.assert_called_once_with('cuda')

    @patch.object(TorchConfigurator, 'wrap_parallel_model')
    def test_create_model_total_parameters(self, mock_wrap_parallel_model):
        # Setup
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [MagicMock(nelement=lambda: 100),
                                              MagicMock(nelement=lambda: 200)]
        mock_task_handler = MagicMock()
        mock_task_handler.create_lm_model.return_value = mock_model
        mock_wrap_parallel_model.return_value = mock_model

        config = MagicMock()
        config.run.compile = False
        config.run.device = 'cuda'

        # Execute
        obj = TorchConfigurator(config, mock_task_handler)
        obj.create_model()

        # Assert
        total_model_parameters = sum([p.nelement() for p in mock_model.parameters()])
        self.assertEqual(total_model_parameters, 300)
        mock_model.to.assert_called_once_with('cuda')


if __name__ == '__main__':
    unittest.main()
