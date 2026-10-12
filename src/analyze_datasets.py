```python
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATHS = {
    "ResNet50": PROJECT_ROOT / "models" / "resnet50_waste.keras",
    "EfficientNetB0": PROJECT_ROOT / "models" / "efficientnetb0_waste.keras"
}

IMAGE_SIZE = (224, 224)

CLASSES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]


def predict_image(image_path, model_name, model_path, image_array):

    if not model_path.exists():
        print(f"\nModel file not found: {model_path}")
        return

    print(f"\nLoading {model_name} model...")

    model = tf.keras.models.load_model(model_path)

    processed_image = image_array.copy()

    if model_name == "ResNet50":
        processed_image = tf.keras.applications.resnet50.preprocess_input(
            processed_image
        )

    predictions = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    print("\n" + "=" * 50)
    print(f"{model_name.upper()} PREDICTION")
    print("=" * 50)
    print(f"Image      : {image_path.name}")
    print(f"Prediction : {CLASSES[predicted_index]}")
    print(f"Confidence : {predictions[predicted_index] * 100:.2f}%")

    print("\nAll class probabilities:")

    for index in np.argsort(predictions)[::-1]:
        print(
            f"{CLASSES[index]:25s}: "
            f"{predictions[index] * 100:.2f}%"
        )

    print("=" * 50)

    del model
    tf.keras.backend.clear_session()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Usage: python src/predict.py "C:\\path\\to\\image.jpg"')
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    for model_name, model_path in MODEL_PATHS.items():
        predict_image(
            image_path,
            model_name,
            model_path,
            image_array
        )
```
