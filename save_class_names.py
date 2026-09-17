from pathlib import Path
import json

# Use exactly the same dataset path as in train_plant_disease.py.
dataset_path = Path(
    r"D:\LeafLens\color"
)

if not dataset_path.is_dir():
    raise FileNotFoundError("Check your color dataset folder path.")

# Match the alphabetical class order used during training.
class_names = sorted(
    folder.name
    for folder in dataset_path.iterdir()
    if folder.is_dir()
)

# Save beside this Python script.
output_path = Path(__file__).parent / "class_names.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

print("Number of classes:", len(class_names))
print("Saved to:", output_path)