"""CNN prediction followed by independent vision assessment and advisory review."""
import argparse
import json
import math
import os
from pathlib import Path
from dataset_config import TRAINING_DATASET_PATH, is_dataset_image
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_PATH = Path(__file__).resolve().parent


class LeafPredictor:
    def __init__(self, model_path=PROJECT_PATH / "leaflens_model.keras",
                 labels_path=PROJECT_PATH / "class_names.json"):

        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        self.tf = tf
        self.class_names = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        if (not isinstance(self.class_names, list) or not self.class_names
                or not all(isinstance(x, str) for x in self.class_names)
                or len(set(self.class_names)) != len(self.class_names)):
            raise ValueError("class_names.json must contain unique class names.")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        if self.model.output_shape[-1] != len(self.class_names):
            raise ValueError("Model output and class_names.json do not match.")

    def predict(self, image_path):
        tf = self.tf
        image = tf.keras.utils.load_img(
            image_path, target_size=(128, 128), interpolation="bilinear")
        # The saved model already includes Rescaling(1 / 255).
        batch = tf.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
        scores = [float(x) for x in self.model.predict(batch, verbose=0)[0]]
        if (not all(math.isfinite(x) and 0 <= x <= 1 for x in scores)
                or not math.isclose(sum(scores), 1, abs_tol=1e-3)):
            raise ValueError("Expected finite softmax probabilities.")
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return {
            "predicted_class": self.class_names[ranked[0]],
            "confidence": scores[ranked[0]],
            "top_predictions": [
                {"class": self.class_names[i], "confidence": scores[i]}
                for i in ranked[:3]
            ],
        }


def combine_results(cnn, vision, review, min_confidence=0.8, errors=None):
    """Only trusted Python code chooses the final class and confidence."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")
    errors = list(errors or [])
    reasons = []
    if cnn["confidence"] < min_confidence:
        reasons.append("low_cnn_confidence")
    if vision is None:
        reasons.append("vision_unavailable")
    elif vision["predicted_class"] != cnn["predicted_class"]:
        reasons.append("vision_uncertain" if vision["predicted_class"] == "uncertain"
                       else "models_disagree")
    if review is None:
        reasons.append("review_unavailable")
    elif review["verdict"] != "supports":
        reasons.append("review_" + review["verdict"])
    if errors:
        reasons.append("ollama_error")
    return {
        "final_class": cnn["predicted_class"],
        "confidence": cnn["confidence"],
        "confidence_note": "CNN softmax score; not calibrated probability or measured accuracy.",
        "status": "needs_review" if reasons else "models_agree",
        "review_reasons": reasons,
        "cnn": cnn,
        "ollama_assessment": vision,
        "ollama_review": review,
        "errors": errors,
    }


def analyze_leaf(image_path, predictor, ollama_model="qwen3-vl:2b",
                 min_confidence=0.8, timeout=120, dataset_path=TRAINING_DATASET_PATH):
    from test_ollama import assess_leaf, review_leaf
    path = Path(image_path).resolve(strict=True)
    cnn = predictor.predict(path)
    if is_dataset_image(path, dataset_path) and cnn["confidence"] >= min_confidence:
        return {
            "final_class": cnn["predicted_class"],
            "confidence": cnn["confidence"],
            "confidence_note": "CNN softmax score; not measured accuracy.",
            "status": "dataset_cnn_only",
            "review_reasons": [],
            "cnn": cnn,
            "ollama_assessment": None,
            "ollama_review": None,
            "errors": [],
        }
    vision = review = None
    errors = []
    try:
        vision = assess_leaf(path, predictor.class_names, ollama_model, timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        errors.append("Independent assessment failed: " + str(exc))
    if vision is not None:
        try:
            review = review_leaf(path, cnn, vision, ollama_model, timeout)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append("Review failed: " + str(exc))
    return combine_results(cnn, vision, review, min_confidence, errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--ollama-model", default="qwen3-vl:2b")
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--cnn-only", action="store_true")
    parser.add_argument("--dataset-path", type=Path, default=TRAINING_DATASET_PATH,
                        help="Dataset root; images inside skip Ollama when confidence meets threshold")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1 or args.timeout <= 0:
        parser.error("Confidence must be in [0, 1] and timeout must be positive.")
    if not args.image.is_file():
        parser.error("Image does not exist.")
    predictor = LeafPredictor()
    result = (predictor.predict(args.image) if args.cnn_only else
              analyze_leaf(args.image, predictor, args.ollama_model,
                           args.min_confidence, args.timeout, args.dataset_path))
    label = result.get("final_class", result.get("predicted_class"))
    status = result.get("status", "CNN only; not reviewed").replace("_", " ")
    print(f"Final result: {label} ({status})")
    print(f"Assessment confidence: {result['confidence']:.2%}")
    #print("Assessment accuracy: not measured (requires labeled test data)")


if __name__ == "__main__":
    main()
