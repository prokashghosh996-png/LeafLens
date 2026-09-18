"""Standalone CNN prediction for debugging leaf classification."""
import argparse
import json
import math
import os
from pathlib import Path
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
                 labels_path=None):

        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        self.tf = tf
        model_path = Path(model_path)
        labels_path = labels_path or model_path.with_name("class_names.json")
        self.class_names = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        if (not isinstance(self.class_names, list) or not self.class_names
                or not all(isinstance(x, str) for x in self.class_names)
                or len(set(self.class_names)) != len(self.class_names)):
            raise ValueError("class_names.json must contain unique class names.")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.image_size = tuple(self.model.input_shape[1:3])
        if not all(isinstance(n, int) and n > 0 for n in self.image_size):
            raise ValueError("Model requires a fixed image size.")
        if self.model.output_shape[-1] != len(self.class_names):
            raise ValueError("Model output and class_names.json do not match.")

    def grad_cam(self, image_path, class_index=None):
        tf = self.tf
        image = tf.keras.utils.load_img(image_path, interpolation="bilinear")
        original_size = (image.height, image.width)
        resized = tf.keras.utils.load_img(
            image_path, target_size=self.image_size, interpolation="bilinear")
        batch = tf.expand_dims(tf.keras.utils.img_to_array(resized), axis=0)

        feature_layer = next(
            (index, layer) for index, layer in reversed(list(enumerate(self.model.layers)))
            if len(layer.output.shape) == 4)
        feature_index, feature_layer = feature_layer
        feature_model = tf.keras.models.Model(self.model.inputs, feature_layer.output)
        classifier_input = tf.keras.Input(shape=feature_layer.output.shape[1:])
        classifier_output = classifier_input
        for layer in self.model.layers[feature_index + 1:]:
            classifier_output = layer(classifier_output)
        classifier_model = tf.keras.models.Model(classifier_input, classifier_output)
        with tf.GradientTape() as tape:
            feature_maps = feature_model(batch, training=False)
            tape.watch(feature_maps)
            predictions = classifier_model(feature_maps, training=False)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            score = predictions[:, class_index]
        gradients = tape.gradient(score, feature_maps)
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return tf.image.resize(heatmap[..., tf.newaxis], original_size)[..., 0].numpy(), image

    def save_grad_cam(self, image_path, output_path, class_index=None):
        import matplotlib.pyplot as plt

        heatmap, image = self.grad_cam(image_path, class_index)
        figure, axis = plt.subplots(figsize=(9, 7))
        axis.imshow(image)
        axis.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axis.axis("off")
        figure.tight_layout(pad=0)
        figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(figure)

    def predict(self, image_path):
        tf = self.tf
        image = tf.keras.utils.load_img(
            image_path, target_size=self.image_size, interpolation="bilinear")
        # Normalization is embedded in both legacy CNN and new MobileNetV2 models.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=PROJECT_PATH / "leaflens_model.keras")
    parser.add_argument("--cnn-only", action="store_true", help="Compatibility option; predictions are always CNN-only")
    parser.add_argument("--heatmap", type=Path,
                        help="Save a Grad-CAM heatmap overlay to this PNG path")
    parser.add_argument("--show-heatmap", action="store_true",
                        help="Display the photo with a Grad-CAM heatmap overlay")
    parser.add_argument("--no-show-image", dest="show_image", action="store_false",
                        help="Print results without opening the photo window")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Image does not exist.")
    predictor = LeafPredictor(model_path=args.model)
    result = predictor.predict(args.image)
    label = result["predicted_class"]
    health_text = result["health_status"].capitalize()
    print(f"CNN predicted class: {label}")
    print(f"CNN health prediction: {health_text} (derived from class label)")
    print(f"CNN class confidence: {result['confidence']:.2%}")
    print("Top predictions:")
    for item in result["top_predictions"]:
        print(f"  {item['class']}: {item['confidence']:.2%}")
    heatmap = None
    photo = None
    if args.heatmap or args.show_heatmap:
        heatmap, pil_image = predictor.grad_cam(
            args.image, class_index=predictor.class_names.index(label))
        if args.heatmap:
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(9, 7))
            axis.imshow(pil_image)
            axis.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axis.axis("off")
            figure.tight_layout(pad=0)
            figure.savefig(args.heatmap, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(figure)
            print(f"Grad-CAM heatmap saved to: {args.heatmap}")
        if args.show_heatmap:
            photo = pil_image
    if args.show_image or args.show_heatmap:
        import matplotlib.pyplot as plt
        if photo is None:
            from PIL import Image
            with Image.open(args.image) as image:
                photo = image.convert("RGB")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(photo)
        if heatmap is not None:
            ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.axis("off")
        heading = (f"Disease status: {health_text} (class-derived)\n"
                   f"{label}\nCNN class confidence: {result['confidence']:.2%}")
        ax.set_title(heading)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
