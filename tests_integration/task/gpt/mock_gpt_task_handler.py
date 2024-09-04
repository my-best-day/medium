from task.gpt.gpt_task_handler import GptTaskHandler
from ...data.gpt.mock_gpt_tokenizer import MockGptTokenizer


class MockGptTaskHandler(GptTaskHandler):

    def create_tokenizer(self):
        tokenizer = MockGptTokenizer()
        return tokenizer
