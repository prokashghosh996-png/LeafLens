"""Train MobileNetV2 using all class folders, with held-out validation and test data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

from dataset_config import TRAINING_DATASET_PATH

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def split_groups(groups, seed):
    """Keep byte-identical images together; split each class approximately 70/15/15."""
    groups = list(groups)
    if len(groups) < 3:
        raise ValueError("Each class needs at least three distinct image groups.")
    random.Random(seed).shuffle(groups)
    held = max(1, int(len(groups) * 0.15))
    partitions = (groups[2 * held:], groups[:held], groups[held:2 * held])
    return {name: [p for group in part for p in group]
            for name, part in zip(("train", "validation", "test"), partitions)}


def inspect_dataset(root, seed):
    from PIL import Image
    labels = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(labels) < 2:
        raise ValueError("Expected at least two class folders.")
    splits = {name: [] for name in ("train", "validation", "test")}
    hashes = {}
    counts = {}
    for index, label in enumerate(labels):
        groups = defaultdict(list)
        for path in sorted((root / label).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            with Image.open(path) as image:
                image.convert("RGB").load()  # Fail on unreadable files; never silently skip.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != label:
                raise ValueError(f"Identical image has conflicting labels: {path}")
            hashes[digest] = label
            groups[digest].append(str(path.relative_to(root)))
        divided = split_groups(groups.values(), seed + index)
        counts[label] = {name: len(items) for name, items in divided.items()}
        for name, items in divided.items():
            splits[name].extend({"path": item, "label": index} for item in items)
        print(f"Inspected {label}: {counts[label]}", flush=True)
    return labels, splits, counts


def build_model(tf, classes, size=224, weights="imagenet"):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights=weights)
    base.trainable = False
    inputs = tf.keras.Input((size, size, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=TRAINING_DATASET_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not args.dataset.is_dir() or args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        parser.error("Dataset must exist; epochs/batch size must be positive and fine-tune epochs nonnegative.")
    root = args.dataset.resolve()
    labels, splits, counts = inspect_dataset(root, args.seed)
    out = args.output or Path(__file__).parent / "training_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    save("class_names.json", labels)
    save("split_manifest.json", {"dataset": str(root), "seed": args.seed,
         "splits": splits, "counts": counts,
         "note": "Exact byte duplicates grouped; near duplicates are not detected."})
    print(f"Artifacts: {out.resolve()}", flush=True)
    if args.inspect_only:
        return

    import numpy as np
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    size = 224

    def dataset(rows, training=False):
        paths = [str(root / row["path"]) for row in rows]
        targets = [row["label"] for row in rows]
        ds = tf.data.Dataset.from_tensor_slices((paths, targets))
        if training:
            ds = ds.shuffle(len(paths), seed=args.seed)
        # Use identical Pillow/Keras resizing in training and inference.
        def read(path):
            image = tf.keras.utils.load_img(path.numpy().decode(), target_size=(size, size),
                                           interpolation="bilinear")
            return tf.keras.utils.img_to_array(image)
        def load(path, target):
            image = tf.py_function(read, [path], tf.float32)
            image.set_shape((size, size, 3))
            return image, target
        return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(1)

    train = dataset(splits["train"], True)
    validation = dataset(splits["validation"])
    test = dataset(splits["test"])
    frequencies = Counter(row["label"] for row in splits["train"])
    class_weights = {i: len(splits["train"]) / (len(labels) * frequencies[i]) for i in range(len(labels))}
    model, base = build_model(tf, len(labels))
    best_path = out / "leaflens_model.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(best_path), monitor="val_loss", save_best_only=True)
    def callbacks():
        return [checkpoint,
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)]
    def compile_model(rate):
        model.compile(optimizer=tf.keras.optimizers.Adam(rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    compile_model(1e-3)
    history = model.fit(train, validation_data=validation, epochs=args.epochs,
                        class_weight=class_weights, callbacks=callbacks())
    save("head_history.json", history.history)
    if args.fine_tune_epochs:
        # EarlyStopping restored the best head-stage weights.
        base.trainable = True
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-30:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        compile_model(1e-5)
        history = model.fit(train, validation_data=validation, epochs=args.fine_tune_epochs,
                            class_weight=class_weights, callbacks=callbacks())
        save("fine_tune_history.json", history.history)
    # The same checkpoint retains the lowest validation loss across both stages.
    best = tf.keras.models.load_model(best_path)
    metrics = best.evaluate(test, return_dict=True, verbose=0)
    scores = best.predict(test, verbose=0)
    truth = np.array([row["label"] for row in splits["test"]])
    predicted = scores.argmax(axis=1)
    matrix = tf.math.confusion_matrix(truth, predicted, num_classes=len(labels)).numpy()
    save("test_metrics.json", {"metrics": metrics, "test_images": len(truth),
         "class_names": labels, "confusion_matrix": matrix.tolist(),
         "per_class_recall": {name: float(matrix[i, i] / matrix[i].sum()) for i, name in enumerate(labels)},
         "note": "Held out from this run only; same-source data does not measure field generalization."})
    save("training_config.json", {"architecture": "MobileNetV2", "image_size": size,
         "seed": args.seed, "batch_size": args.batch_size, "head_epochs": args.epochs,
         "fine_tune_epochs": args.fine_tune_epochs, "class_weights": class_weights})
    print(json.dumps(metrics, indent=2))
    print(f'Predict with: python predict_leaf.py "IMAGE_PATH" --model "{best_path.resolve()}"')


if __name__ == "__main__":
    main()
