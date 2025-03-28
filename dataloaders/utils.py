from typing import Tuple

from torchvision import transforms


REQUIRED_IMG_SIZE = (512, 512)

BRAIN_MRI_TRANSFORMS = [
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
]

BREAST_CT_TRANSFORMS = [
    transforms.RandomRotation(degrees=20),
    transforms.RandomErasing(p=0.2),
]


LUNGS_XRAY_TRANSFORMS = [
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, shear=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomErasing(p=0.3),
]

TRANSFORM_MAP = {
    "BRAIN": BRAIN_MRI_TRANSFORMS,
    "BREAST": BREAST_CT_TRANSFORMS,
    "LUNGS": LUNGS_XRAY_TRANSFORMS,
}


def get_transform(
    scanned_organ: str, mode: str, image_size: Tuple[int, int] = (512, 512)
):
    needs_resizing = False
    if image_size != REQUIRED_IMG_SIZE:
        needs_resizing = True

    pipeline = [transforms.Grayscale(num_output_channels=1)]

    if needs_resizing:
        pipeline.append(
            transforms.Resize(REQUIRED_IMG_SIZE)
        )  # Resize before converting to tensor

    if mode.upper() == "TRAIN":
        pipeline += TRANSFORM_MAP[scanned_organ.upper()]

    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize((0.5,), (0.5,)))  # Add normalization last

    return transforms.Compose(pipeline)
