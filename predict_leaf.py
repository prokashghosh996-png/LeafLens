"""CNN prediction followed by independent vision assessment and advisory review."""
import argparse
import json
import math
import os
from pathlib import Path
from dataset_config import TRAINING_DATASET_PATH, is_dataset_image
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_PATH = Path(__file__).resolve().parent


def health_from_class(label):
    """Map dataset class labels to health without using the input folder as truth."""
    if "___" not in label:
        return "uncertain"
    condition = label.rsplit("___", 1)[1].strip()
    if not condition:
        return "uncertain"
    return "not diseased" if condition.lower() == "healthy" else "diseased"

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
            "health_status": health_from_class(self.class_names[ranked[0]]),
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
    if vision is not None:
        vision_health = vision.get("health_status", "uncertain")
        if vision_health == "uncertain":
            reasons.append("vision_health_uncertain")
        elif vision_health != health_from_class(cnn["predicted_class"]):
            reasons.append("health_assessments_disagree")
    if review is None:
        reasons.append("review_unavailable")
    elif review["verdict"] != "supports":
        reasons.append("review_" + review["verdict"])
    if review is not None and vision is not None:
        if review.get("health_status", "uncertain") != vision.get("health_status", "uncertain"):
            reasons.append("visual_reviews_disagree")
    if errors:
        reasons.append("ollama_error")
    return {
        "final_class": cnn["predicted_class"],
        "health_status": ("uncertain" if reasons else vision_health),
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
    parser.add_argument("--no-show-image", dest="show_image", action="store_false", help="Print results without opening the photo window")
    parser.add_argument("--dataset-path", type=Path, default=TRAINING_DATASET_PATH,
                        help="Legacy dataset path option; visual review now runs for all images")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1 or args.timeout <= 0:
        parser.error("Confidence must be in [0, 1] and timeout must be positive.")
    if not args.image.is_file():
        parser.error("Image does not exist.")
    predictor = LeafPredictor()
    result = (combine_results(predictor.predict(args.image), None, None, args.min_confidence) if args.cnn_only else
              analyze_leaf(args.image, predictor, args.ollama_model,
                           args.min_confidence, args.timeout, args.dataset_path))
    label = result.get("final_class", result.get("predicted_class"))
    status = result.get("status", "CNN only; not reviewed").replace("_", " ")
    health_text = {"diseased": "Diseased", "not diseased": "No visible disease detected",
                   "uncertain": "Uncertain - needs review"}[result["health_status"]]
    print(f"Final result: {health_text}")
    print(f"CNN predicted class (supporting information): {label}")
    print(f"CNN class confidence: {result['confidence']:.2%}")
    vision = result.get("ollama_assessment")
    if vision is not None:
        print("Ollama assessment:", vision.get("health_status", "uncertain"))
        print("Ollama observations:", vision["observations"])
    else:
        print("Ollama assessment:", "unavailable" if result.get("errors") else "not requested")
    if result.get("ollama_review"):
        print("Ollama review:", result["ollama_review"]["reason"])
    for error in result.get("errors", []):
        print("Ollama error:", error)
    if result.get("review_reasons"):
        print("Review reasons:", ", ".join(result["review_reasons"]))
    if args.show_image:
        import matplotlib.pyplot as plt
        from PIL import Image
        with Image.open(args.image) as image:
            photo = image.convert("RGB")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(photo)
        ax.axis("off")
        heading = (f"{health_text} | {status}\n"
                   f"{label}\nCNN class confidence: {result['confidence']:.2%}")
        if vision is not None:
            heading += "\nOllama: " + vision.get("health_status", "uncertain")
        else:
            heading += "\nOllama: " + ("unavailable" if result.get("errors") else "not requested")
        ax.set_title(heading)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
