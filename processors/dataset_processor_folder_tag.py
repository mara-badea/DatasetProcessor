import os
import pandas as pd
from processors.dataset_processor import DatasetProcessor
import os


SCAN_TYPE_DICT = {"rmn": 0, "mri": 0, "xray": 1, "ct": 2}

BRAIN_DATASET_DISEASE_DICT = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3,
}

LUNGS_DATASET_DISEASE_DICT = {"normal": 0, "pneumonia": 4}


import numpy as np


class DatasetProcessorFolderTag(DatasetProcessor):
    def __init__(
        self,
        dataset_path: str,
        scanned_organ: str,
        scan_type: str,
        csv_dest_folder: str,
        csv_file_name: str,
    ):
        super().__init__(dataset_path, csv_dest_folder, csv_file_name)
        self.organ_label = 0 if scanned_organ.lower() == "brain" else 1
        self.scan_label = SCAN_TYPE_DICT[scan_type.lower()]
        self.disease_dict = (
            BRAIN_DATASET_DISEASE_DICT
            if self.organ_label == 0
            else LUNGS_DATASET_DISEASE_DICT
        )
        self.train_disease_label = self._get_disease_label(
            self.train_set_path, self.disease_dict
        )
        self.test_disease_label = self._get_disease_label(
            self.test_set_path, self.disease_dict
        )

        # Store labels and metadata in memory
        self.all_labels = {
            "train": self._process_labels(self.train_disease_label),  # 0 for train
            "test": self._process_labels(self.test_disease_label),  # 1 for test
        }

        # Create CSV from memory data
        self.create_csv(self.all_labels["train"], dataset_type=0)  # 0 for train
        self.create_csv(self.all_labels["test"], dataset_type=1)  # 1 for test

    def _process_labels(self, disease_labels: dict):
        """
        Prepare labels by including all necessary metadata: image path, disease label, organ label, scan type label, dataset type.
        Convert disease, organ, and scan labels to one-hot encoded format.
        """
        labels_with_metadata = {}
        for img_path, disease_label in disease_labels.items():
            labels_with_metadata[img_path] = {
                "disease_label": self.one_hot_encode(
                    disease_label, 8
                ),  # 8 possible diseases
                "organ_label": self.one_hot_encode(
                    self.organ_label, 3
                ),  # 3 possible organs
                "scan_label": self.one_hot_encode(
                    self.scan_label, 3
                ),  # 3 possible scan types
            }
        return labels_with_metadata

    def one_hot_encode(self, label, num_classes):
        """
        Converts a label to a one-hot encoded vector.
        """
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[label] = 1.0
        return one_hot

    def create_csv(self, disease_labels_with_metadata: dict, dataset_type: int):
        saved_csv_file_name = (
            f"test_{self.csv_file_name}"
            if dataset_type == 0
            else f"train_{self.csv_file_name}"
        )
        csv_path = os.path.join(self.csv_dest_folder, saved_csv_file_name)
        self._ensure_directory_exists(csv_path)

        rows = [
            {
                "Image Path": img_path,
                "Organ Label": metadata["organ_label"].tolist(),
                "Disease Label": metadata["disease_label"].tolist(),
                "Scan Type Label": metadata["scan_label"].tolist(),
            }
            for img_path, metadata in disease_labels_with_metadata.items()
        ]

        df = pd.DataFrame(rows)

        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", index=False, header=write_header)

    @staticmethod
    def _ensure_directory_exists(path: str):
        """Ensure the directory for the CSV file exists."""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_labels(self):
        return self.all_labels

    @staticmethod
    def _get_disease_label(set_path: str, disease_dict: dict):
        image_labels = {}
        data_folder = os.listdir(set_path)

        for tag in data_folder:
            image_folder_path = os.path.join(set_path, tag)
            images = os.listdir(image_folder_path)

            for image in images:
                image_path = os.path.join(image_folder_path, image)
                image_labels[image_path] = disease_dict[tag.lower()]

        return image_labels
