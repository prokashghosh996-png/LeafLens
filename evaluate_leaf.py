"""Measure labeled-folder accuracy. Use a genuinely held-out dataset."""
import argparse
import json
import random
import sys
from pathlib import Path
from predict_leaf import LeafPredictor, analyze_leaf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Folder containing class-name subfolders")
    parser.add_argument("--with-ollama", action="store_true",
                        help="Also measure independent vision accuracy and review coverage")
    parser.add_argument("--limit", type=int, help="Seeded random sample size")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ollama-model", default="qwen3-vl:2b")
    args = parser.parse_args()
    if not args.dataset.is_dir() or (args.limit is not None and args.limit <= 0):
        parser.error("Dataset must exist and limit must be positive.")
    predictor = LeafPredictor()
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
    correct = vision_correct = vision_available = agreed = agreed_correct = 0
    per_class = {}
    for index, path in enumerate(paths, 1):
        if args.with_ollama:
            result = analyze_leaf(path, predictor, args.ollama_model)
            cnn = result["cnn"]
            vision = result["ollama_assessment"]
            vision_available += vision is not None
            vision_correct += vision is not None and vision["predicted_class"] == path.parent.name
            if result["status"] == "models_agree":
                agreed += 1
                agreed_correct += result["final_class"] == path.parent.name
        else:
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
    if args.with_ollama:
        report.update({
            "final_accuracy": correct / len(paths),
            "vision_accuracy_all_images": vision_correct / len(paths),
            "vision_available": vision_available,
            "vision_accuracy_when_available": vision_correct / vision_available if vision_available else None,
            "agreement_coverage": agreed / len(paths),
            "accuracy_on_agreed_images": agreed_correct / agreed if agreed else None,
            "needs_review": len(paths) - agreed,
            "policy": "Final labels are locked to CNN, so final accuracy equals CNN accuracy.",
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
