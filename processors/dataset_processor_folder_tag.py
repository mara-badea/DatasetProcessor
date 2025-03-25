import os
import csv

from dataset_processor import DatasetProcessor

SCAN_TYPE_DICT = {
    "rmn": 0,
    "mri": 0,
    "xray": 1,
    "ct": 2
}

BRAIN_DATASET_DISEASE_DICT = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3
}

LUNG_DATASET_DISEASE_DICT = {}


class DatasetProcessorFolderTag(DatasetProcessor):
    def __init__(
            self, dataset_path: str, scanned_organ: str, scan_type: str, output_csv_path: str
    ):
        super().__init__(dataset_path, output_csv_path)
        self.organ_label = 0 if scanned_organ.lower() == "brain" else 1
        self.scan_label = SCAN_TYPE_DICT[scan_type.lower()]
        self.disease_dict = BRAIN_DATASET_DISEASE_DICT if self.organ_label == 0 else LUNG_DATASET_DISEASE_DICT
        self.train_disease_label = self._get_disease_label(self.train_set_path, self.disease_dict)
        self.test_disease_label = self._get_disease_label(self.test_set_path, self.disease_dict)
        self._create_csv(self.train_disease_label, dataset_type=0)  # 1 for train
        self._create_csv(self.test_disease_label, dataset_type=1)  # 0 for test

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

    def _create_csv(self, disease_labels: dict, dataset_type: int):
        """
        Create a CSV file with image paths, disease labels, and scan type labels.
        The dataset_type will be 0 for train and 1 for test.
        """
        # Open the CSV file in append mode to add data for both train and test sets
        with open(self.output_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Image Path', 'Scanned Organ', 'Disease Label', 'Scan Type Label', 'Dataset Type'])

            for img_path, disease_label in disease_labels.items():
                 writer.writerow([img_path, self.organ_label, disease_label, self.scan_label, dataset_type])
