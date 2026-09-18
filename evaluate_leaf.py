"""Measure labeled-folder accuracy. Use a genuinely held-out dataset."""
import argparse
import json
import random
import sys
from pathlib import Path
from predict_leaf import LeafPredictor, PROJECT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Folder containing class-name subfolders")
    parser.add_argument("--limit", type=int, help="Seeded random sample size")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", type=Path, default=PROJECT_PATH / "leaflens_model.keras")
    args = parser.parse_args()
    if not args.dataset.is_dir() or (args.limit is not None and args.limit <= 0):
        parser.error("Dataset must exist and limit must be positive.")
    predictor = LeafPredictor(model_path=args.model)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    paths = sorted(p for p in args.dataset.rglob("*")
                   if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        parser.error("No supported images found.")
    unknown = sorted({p.parent.name for p in paths} - set(predictor.class_names))
    if unknown:
        parser.error("Unknown class folders: " + ", ".join(unknown))
    if args.limit is not None:
        paths = random.Random(args.seed).sample(paths, min(args.limit, len(paths)))
    correct = 0
    per_class = {}
    for index, path in enumerate(paths, 1):
        cnn = predictor.predict(path)
        hit = cnn["predicted_class"] == path.parent.name
        correct += hit
        counts = per_class.setdefault(path.parent.name, {"correct": 0, "total": 0})
        counts["correct"] += int(hit)
        counts["total"] += 1
        print(f"Evaluated {index}/{len(paths)}", file=sys.stderr)
    report = {
        "images": len(paths),
        "cnn_accuracy": correct / len(paths),
        "per_class": per_class,
        "note": "Accuracy on supplied images only. Training overlap has not been verified; "
                "do not claim held-out accuracy unless these images were excluded from training.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
