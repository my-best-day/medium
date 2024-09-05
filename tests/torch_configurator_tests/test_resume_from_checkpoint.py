import unittest
from unittest.mock import patch, MagicMock
from torch_configurator import TorchConfigurator


class TestResumeFromCheckpoint(unittest.TestCase):
    """
    Tests that checkpoint is loaded and resume_from_checkpoint is called if is_primary,
    nothing if not primary.
    """

    def setUp(self):
        self.config = MagicMock()
        self.config.run = MagicMock()
        self.config.train = MagicMock()
        self.config.train.checkpoint = 'path/to/checkpoint'
        self.config.run.device = 'cpu'
        self.mock_task_handler = MagicMock()

    @patch('torch_configurator.TorchConfigurator.validate_checkpoint_path')
    @patch('torch.load')
    def test_resume_from_checkpoint_primary(self, mock_load, mock_validate_checkpoint_path):
        self.config.run.is_primary = True
        mock_load.return_value = {'model_state_dict': 'model_state_dict'}
        mock_validate_checkpoint_path.return_value = True

        configurator = TorchConfigurator(self.config, self.mock_task_handler)
        configurator.model = MagicMock()
        configurator.trainer = MagicMock()
        configurator.optimizer = MagicMock()

        configurator.resume_from_checkpoint()

        mock_load.assert_called_once_with(self.config.train.checkpoint, map_location='cpu',
                                          weights_only=False)
        configurator.task_handler.resume_from_checkpoint_dict.assert_called_once_with(
            configurator.model, configurator.optimizer, configurator.trainer,
            {'model_state_dict': 'model_state_dict'})

    @patch('torch_configurator.TorchConfigurator.validate_checkpoint_path')
    @patch('torch.load')
    def test_resume_from_checkpoint_not_primary(self, mock_load, mock_validate_checkpoint_path):
        self.config.run.is_primary = False
        mock_load.return_value = {'model_state_dict': 'model_state_dict'}
        mock_validate_checkpoint_path.return_value = True

        configurator = TorchConfigurator(self.config, self.mock_task_handler)
        configurator.model = MagicMock()
        configurator.trainer = MagicMock()
        configurator.optimizer = MagicMock()

        configurator.resume_from_checkpoint()

        mock_load.assert_not_called()
