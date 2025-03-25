import os


class DatasetProcessor:
    def __init__(self, dataset_path: str, output_csv_path: str):
        self.dataset_path = dataset_path
        self.output_csv_path = output_csv_path
        self.train_set_path = self._get_train_set_path()
        self.test_set_path = self._get_test_set_path()

    def _get_train_set_path(self):
        folders = os.listdir(self.dataset_path)

        for folder in folders:
            if "train" in folder.lower():
                return os.path.join(self.dataset_path, folder)

        raise Exception("No train folder found!")

    def _get_test_set_path(self):
        folders = os.listdir(self.dataset_path)

        for folder in folders:
            if "test" in folder.lower():
                return os.path.join(self.dataset_path, folder)

        raise Exception("No test folder found!")