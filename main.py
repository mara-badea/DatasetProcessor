import numpy as np
import torchvision
from matplotlib import pyplot as plt

from dataloaders.custom_data_loader import CustomDataLoader

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


dataset_path = r"C:\Users\badea\Downloads\DatasetProcessor\data\brain-tumor-mri-dataset"
output_csv_path = r"C:\Users\badea\Downloads\DatasetProcessor\output\brain-labels.csv"

custom_data_loader = CustomDataLoader(dataset_path, "brain", "mri", output_csv_path)

N = 2

augmented_images = []

for _ in range(N):
    for image, label in custom_data_loader.dataset:
        augmented_images.append((image, label))

print(f"Total augmented images generated: {len(augmented_images)}")


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random training images

dataiter = iter(custom_data_loader.dataset)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
