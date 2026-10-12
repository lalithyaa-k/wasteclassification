
from pathlib import Path
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "processed" / "dataset" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetb0_waste.keras"
RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

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

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("\nLoading EfficientNetB0...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating test dataset...")
start_time = time.time()

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_dataset
])

probabilities = model.predict(test_dataset, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

inference_time = time.time() - start_time

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_true, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASSES)),
    target_names=CLASSES,
    zero_division=0
)

print("\n" + "=" * 55)
print("EFFICIENTNETB0 TEST RESULTS")
print("=" * 55)
print(f"Test accuracy:       {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Weighted precision:  {precision:.4f}")
print(f"Weighted recall:     {recall:.4f}")
print(f"Weighted F1-score:   {f1:.4f}")
print(f"Inference time:      {inference_time:.2f} seconds")

print("\nCLASSIFICATION REPORT")
print(report)

report_path = RESULTS_PATH / "efficientnetb0_classification_report.txt"
with open(report_path, "w", encoding="utf-8") as file:
    file.write("EFFICIENTNETB0 TEST RESULTS\n\n")
    file.write(f"Test accuracy: {accuracy:.4f}\n")
    file.write(f"Weighted precision: {precision:.4f}\n")
    file.write(f"Weighted recall: {recall:.4f}\n")
    file.write(f"Weighted F1-score: {f1:.4f}\n\n")
    file.write(report)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASSES))
)

fig, ax = plt.subplots(figsize=(12, 10))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
    colorbar=False
)

ax.set_title("EfficientNetB0 - Test Confusion Matrix")
fig.tight_layout()

cm_path = RESULTS_PATH / "efficientnetb0_confusion_matrix.png"
fig.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close(fig)

results = {
    "model": "EfficientNetB0",
    "test_samples": int(len(y_true)),
    "test_accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1_score": float(f1),
    "inference_time_seconds": round(inference_time, 2)
}

results_path = RESULTS_PATH / "efficientnetb0_test_results.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nFiles saved:")
print(report_path)
print(cm_path)
print(results_path)
print("\nEvaluation complete.")
