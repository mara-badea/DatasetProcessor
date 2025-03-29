from torch.utils.data import DataLoader
from custom_dataloaders.utils import get_transform
from custom_datasets.custom_image_dataset import CustomImageDataset


from processors.dataset_processor_folder_tag import DatasetProcessorFolderTag


class CustomDataLoader:
    def __init__(
        self,
        dataset_path: str,
        scanned_organ: str,
        scan_type: str,
        csv_dest_folder: str,
        csv_file_name: str,
        batch_size: int = 32,
        mode: str = "train",  # Default mode is train
    ):
        self.transforms = get_transform(scanned_organ, mode)
        self.dataset_processor = (
            DatasetProcessorFolderTag(
                dataset_path, scanned_organ, scan_type, csv_dest_folder, csv_file_name
            )
            if scanned_organ.upper() in {"BRAIN", "LUNGS"}
            else None
        )
        self.mode = mode

        # Load the custom dataset with the appropriate transformations
        self.dataset = self.load_custom_dataset()

        # Create data loaders for different custom_datasets
        self.train_loader = (
            DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
            if mode == "train"
            else None
        )
        self.val_loader = (
            DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            if mode in ["val", "test"]
            else None
        )

    def load_custom_dataset(self):
        # Get image paths from the dataset directory
        labels = self.dataset_processor.get_labels()[self.mode.lower()]
        image_paths = [img_path for img_path in labels]

        # Get custom labels from the processor
        custom_labels = [labels[img_path] for img_path in labels]

        # Use CustomImageDataset to load images and labels with transforms
        custom_dataset = CustomImageDataset(image_paths, custom_labels, self.transforms)

        return custom_dataset
