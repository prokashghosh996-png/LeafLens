"""Dataset location shared by training and prediction."""
from pathlib import Path

TRAINING_DATASET_PATH = Path(__file__).resolve().parent / "color"


def is_dataset_image(image_path, dataset_path=TRAINING_DATASET_PATH):
    """Check resolved directory membership, not filename or string-prefix equality."""
    image = Path(image_path).resolve()
    dataset = Path(dataset_path).resolve()
    return image.is_file() and image.is_relative_to(dataset)
