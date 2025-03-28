import os
from abc import ABC, abstractmethod


class DatasetProcessor(ABC):
    def __init__(self, dataset_path: str, csv_dest_folder: str, csv_file_name: str):
        self.dataset_path = dataset_path
        self.csv_dest_folder = csv_dest_folder
        self.csv_file_name = csv_file_name
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

    @abstractmethod
    def get_labels(self):
        """
        Abstract method to retrieve labels for the dataset.
        Should be implemented by subclasses to handle custom label logic.
        """
        pass
