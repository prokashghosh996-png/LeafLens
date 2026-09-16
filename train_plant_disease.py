import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path

from dataset_config import TRAINING_DATASET_PATH

dataset_path = TRAINING_DATASET_PATH

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    label_mode="int"
)

class_names = train_ds.class_names

plt.figure(figsize=(12, 10))

for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])], fontsize=8)
        plt.axis("off")

#plt.tight_layout()
#plt.show()
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
for images, labels in train_ds.take(1):
    normalized_images = normalization_layer(images)

    print("Before:", float(tf.reduce_min(images)),
          float(tf.reduce_max(images)))

    print("After:", float(tf.reduce_min(normalized_images)),
          float(tf.reduce_max(normalized_images)))

    #Building CNN
model = tf.keras.Sequential([
tf.keras.Input(shape=(128, 128, 3)),
normalization_layer,

tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
])

model.add(tf.keras.layers.Conv2D(
    32, (3, 3), activation="relu"
))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(
    64, (3, 3), activation="relu"
))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(
    len(class_names), activation="softmax"
))

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("leaflens_model.keras")