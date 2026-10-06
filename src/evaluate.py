from pathlib import Path

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "processed" / "dataset" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2_waste.keras"
RESULTS_PATH = PROJECT_ROOT / "results"


# ============================================================
# SETTINGS
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("MOBILENETV2 TEST SET EVALUATION")
print("=" * 60)

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading TEST dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    class_names=CLASSES,
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MOBILENETV2 PREPROCESSING
# ============================================================

def preprocess(images, labels):
    images = tf.cast(images, tf.float32)

    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    )

    return images, labels


test_dataset = test_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\nEvaluating on untouched TEST set...")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


# ============================================================
# GET PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"\nTest Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

report_path = RESULTS_PATH / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("MOBILENETV2 WASTE CLASSIFICATION\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Test Loss     : {test_loss:.4f}\n")
    file.write(f"Test Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision     : {precision:.4f}\n")
    file.write(f"Recall        : {recall:.4f}\n")
    file.write(f"F1-Score      : {f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write("=" * 60 + "\n")
    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(12, 10)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("MobileNetV2 Waste Classification - Confusion Matrix")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

cm_path = RESULTS_PATH / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "test_loss": float(test_loss),
    "test_accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1": float(f1),
    "number_of_test_images": int(len(y_true))
}

summary_path = RESULTS_PATH / "test_results.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("\nSaved files:")

print(f"Classification report:")
print(report_path)

print(f"\nConfusion matrix:")
print(cm_path)

print(f"\nTest results:")
print(summary_path)