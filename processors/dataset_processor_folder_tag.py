import os
import pandas as pd
from processors.dataset_processor import DatasetProcessor


SCAN_TYPE_DICT = {"rmn": 0, "mri": 0, "xray": 1, "ct": 2}

BRAIN_DATASET_DISEASE_DICT = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3,
}

LUNGS_DATASET_DISEASE_DICT = {"normal": 4, "pneumonia": 5}


class DatasetProcessorFolderTag(DatasetProcessor):
    def __init__(
        self,
        dataset_path: str,
        scanned_organ: str,
        scan_type: str,
        output_csv_path: str,
    ):
        super().__init__(dataset_path, output_csv_path)
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
            "train": self._process_labels(
                self.train_disease_label, dataset_type=0
            ),  # 0 for train
            "test": self._process_labels(
                self.test_disease_label, dataset_type=1
            ),  # 1 for test
        }

        # Create CSV from memory data
        self.create_csv(self.all_labels["train"], dataset_type=0)  # 1 for train
        self.create_csv(self.all_labels["test"], dataset_type=1)  # 0 for test

    def _get_disease_label(self, set_path: str, disease_dict: dict):
        image_labels = {}
        data_folder = os.listdir(set_path)

        for tag in data_folder:
            image_folder_path = os.path.join(set_path, tag)
            images = os.listdir(image_folder_path)

            for image in images:
                image_path = os.path.join(image_folder_path, image)
                image_labels[image_path] = disease_dict[tag.lower()]

        return image_labels

    def _process_labels(self, disease_labels: dict, dataset_type: int):
        """
        Prepare labels by including all necessary metadata: image path, disease label, organ label, scan type label, dataset type.
        """
        labels_with_metadata = {}
        for img_path, disease_label in disease_labels.items():
            labels_with_metadata[img_path] = {
                "disease_label": disease_label,
                "organ_label": self.organ_label,
                "scan_label": self.scan_label,
                "dataset_type": dataset_type,
            }
        return labels_with_metadata

    def create_csv(self, disease_labels_with_metadata: dict, dataset_type: int):
        """
        Create a CSV file using pandas with image paths and metadata.
        """
        self._ensure_directory_exists(self.output_csv_path)

        rows = [
            {
                "Image Path": img_path,
                "Organ Label": metadata["organ_label"],
                "Disease Label": metadata["disease_label"],
                "Scan Type Label": metadata["scan_label"],
                "Dataset Type": metadata["dataset_type"],
            }
            for img_path, metadata in disease_labels_with_metadata.items()
        ]

        df = pd.DataFrame(rows)

        write_header = not os.path.exists(self.output_csv_path)
        df.to_csv(self.output_csv_path, mode="a", index=False, header=write_header)

    @staticmethod
    def _ensure_directory_exists(path: str):
        """Ensure the directory for the CSV file exists."""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_labels(self):
        """
        Returns the labels and metadata (organ_label, scan_label, etc.) in memory.
        """
        return self.all_labels
