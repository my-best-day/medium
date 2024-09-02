from task.mlm.bert_mlm_dataset_precached import BertMlmDatasetPrecached


class MockBertMlmDatasetPrecached(BertMlmDatasetPrecached):
    def __init__(self, path, percentage=1.0):
        super().__init__(path, percentage)

    def load_data(self, path):
        return super().load_data(path)

    def __len__(self):
        return super().__len__()
