from pathlib import Path

import sys
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2_waste.keras"


# ============================================================
# SETTINGS
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"\nERROR: Image not found:")
        print(image_path)
        return

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = CLASSES[predicted_index]

    confidence = probabilities[predicted_index] * 100

    # ========================================================
    # RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("WASTE CLASSIFICATION RESULT")
    print("=" * 60)

    print(f"\nImage      : {image_path.name}")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")

    print("\nAll class probabilities:")

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    for index in sorted_indices:
        print(
            f"{CLASSES[index]:25s}: "
            f"{probabilities[index] * 100:.2f}%"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print("python src\\predict.py <image_path>")

        print("\nExample:")
        print(
            'python src\\predict.py "C:\\Users\\lalit\\Pictures\\bottle.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict_image(image_path)