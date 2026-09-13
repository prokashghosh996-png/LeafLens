from pathlib import Path
import json
import tensorflow as tf

project_path = Path(__file__).resolve().parent
model = tf.keras.models.load_model(
    project_path / "leaflens_model.keras",
    compile = False
)

with (project_path / "class_names.json").open(
    "r" , encoding="utf-8"
) as file:
    class_names = json.load(file)


# Replace this with the path with test image path.
image_path = Path(r"D:\Download\leaf-image-procession\leaf-image-procession\archive\plantvillage dataset\color\Tomato___healthy\337e35e3-7b0d-4760-a1f6-c26c7db90a0b___GH_HL Leaf 425.JPG")

image = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128),
    interpolation="bilinear"
)

image_array = tf.keras.utils.img_to_array(image)
image_batch = tf.expand_dims(image_array, axis=0)

scores = model.predict(image_batch, verbose=0)[0]

predicted_id = int(tf.argmax(scores))
predicted_class = class_names[predicted_id]

print("Predicted class:", predicted_class)
print(f"Model score: {scores[predicted_id]:.2%}")