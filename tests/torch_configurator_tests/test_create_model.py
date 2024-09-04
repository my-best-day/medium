import unittest
from unittest.mock import patch, MagicMock
from torch_configurator import TorchConfigurator


class TestCreateModel(unittest.TestCase):

    def setUp(self):
        # Example configuration structure for testing
        self.config = MagicMock()
        self.config.run = MagicMock()
        self.config.run.local_rank = 0  # example rank for DDP
        self.mock_model = MagicMock()
        self.mock_model.to.return_value = self.mock_model
        self.task_handler = MagicMock()
        self.task_handler.create_lm_model.return_value = self.mock_model

    @patch.object(TorchConfigurator, 'wrap_parallel_model')
    @patch('torch.compile', return_value=MagicMock())
    def test_create_model_no_compile(self, mock_compile,
                                     mock_wrap_parallel_model):
        # Setup
        self.task_handler.create_lm_model.return_value = self.mock_model
        mock_wrap_parallel_model.return_value = self.mock_model

        self.config.run.compile = False
        self.config.run.device = 'cuda'

        # Execute
        obj = TorchConfigurator(self.config, self.task_handler)
        model = obj.create_model()

        # Assert
        self.task_handler.create_lm_model.assert_called_once()
        mock_compile.assert_not_called()
        mock_wrap_parallel_model.assert_called_once_with(self.mock_model)
        self.assertEqual(model, mock_wrap_parallel_model.return_value)
        self.mock_model.to.assert_called_once_with('cuda')

    @patch.object(TorchConfigurator, 'wrap_parallel_model')
    def test_create_model_with_compile(self, mock_wrap_parallel_model):
        # Setup
        self.config.run.compile = True
        self.config.run.device = 'cuda'

        # Execute
        obj = TorchConfigurator(self.config, self.task_handler)

        with patch('torch.compile', return_value=self.mock_model) as mock_compile:
            model = obj.create_model()

            # Assert
            self.task_handler.create_lm_model.assert_called_once()
            mock_compile.assert_called_once_with(self.mock_model)
            mock_wrap_parallel_model.assert_called_once_with(mock_compile.return_value)
            self.assertEqual(model, mock_wrap_parallel_model.return_value)
            self.mock_model.to.assert_called_once_with('cuda')

    @patch.object(TorchConfigurator, 'wrap_parallel_model')
    def test_create_model_total_parameters(self, mock_wrap_parallel_model):
        # Setup
        self.mock_model.parameters.return_value = [MagicMock(nelement=lambda: 100),
                                                   MagicMock(nelement=lambda: 200)]
        self.task_handler.create_lm_model.return_value = self.mock_model
        mock_wrap_parallel_model.return_value = self.mock_model

        self.config.run.compile = False
        self.config.run.device = 'cuda'

        # Execute
        obj = TorchConfigurator(self.config, self.task_handler)
        obj.create_model()

        # Assert
        total_model_parameters = sum([p.nelement() for p in self.mock_model.parameters()])
        self.assertEqual(total_model_parameters, 300)
        self.mock_model.to.assert_called_once_with('cuda')


if __name__ == '__main__':
    unittest.main()
