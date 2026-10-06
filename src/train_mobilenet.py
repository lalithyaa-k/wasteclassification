from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "processed"
    / "dataset"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenetv2_waste.keras"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
)


RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 32

SEED = 42

NUM_CLASSES = 9

EPOCHS_PHASE1 = 5

EPOCHS_PHASE2 = 5


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


# ==================================================
# LOAD DATASETS
# ==================================================

print("\nLoading datasets...")


train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH / "train",
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH / "val",
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nClass names:")
print(CLASSES)


# ==================================================
# DATA PREPROCESSING
# ==================================================

def preprocess_dataset(images, labels):

    images = tf.cast(
        images,
        tf.float32
    )

    images = preprocess_input(
        images
    )

    return images, labels


train_dataset = train_dataset.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==================================================
# CALCULATE CLASS WEIGHTS
# ==================================================

print("\nCalculating class weights...")


class_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)


for _, labels in train_dataset:

    labels = labels.numpy()

    for label in labels:

        class_counts[label] += 1


classes_array = np.arange(
    NUM_CLASSES
)


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes_array,
    y=np.repeat(
        classes_array,
        class_counts
    )
)


class_weights = {
    i: float(class_weights_array[i])
    for i in range(NUM_CLASSES)
}


print("\nClass distribution:")

for i, class_name in enumerate(CLASSES):

    print(
        f"{class_name:25} : "
        f"{class_counts[i]:6d} "
        f"| weight = "
        f"{class_weights[i]:.4f}"
    )


# ==================================================
# BUILD MOBILENETV2
# ==================================================

print("\nBuilding MobileNetV2...")


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(
        224,
        224,
        3
    )
)


# Phase 1: freeze base model

base_model.trainable = False


inputs = layers.Input(
    shape=(
        224,
        224,
        3
    )
)


x = base_model(
    inputs,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ==================================================
# PHASE 1
# ==================================================

print("\n========================================")
print("PHASE 1: FEATURE EXTRACTION")
print("========================================")


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


checkpoint_phase1 = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping_phase1 = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)


reduce_lr_phase1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=[
        checkpoint_phase1,
        early_stopping_phase1,
        reduce_lr_phase1
    ]
)


# ==================================================
# PHASE 2
# ==================================================

print("\n========================================")
print("PHASE 2: FINE-TUNING")
print("========================================")


base_model.trainable = True


# Freeze lower layers
# Fine-tune upper layers only

fine_tune_at = 100


for layer in base_model.layers[
    :fine_tune_at
]:

    layer.trainable = False


print(
    f"\nTotal MobileNetV2 layers: "
    f"{len(base_model.layers)}"
)

print(
    f"Fine-tuning from layer: "
    f"{fine_tune_at}"
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


checkpoint_phase2 = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping_phase2 = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


reduce_lr_phase2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=[
        checkpoint_phase2,
        early_stopping_phase2,
        reduce_lr_phase2
    ]
)


# ==================================================
# SAVE FINAL MODEL
# ==================================================

model.save(
    MODEL_PATH
)


# ==================================================
# SAVE CLASS NAMES
# ==================================================

class_file = (
    RESULTS_PATH
    / "class_names.json"
)


with open(
    class_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        CLASSES,
        f,
        indent=4
    )


# ==================================================
# FINAL VALIDATION
# ==================================================

loss, accuracy = model.evaluate(
    val_dataset,
    verbose=1
)


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Validation Loss     : {loss:.4f}"
)

print(
    f"Validation Accuracy : {accuracy:.4f}"
)

print(
    f"\nModel saved to:"
)

print(MODEL_PATH)

print(
    f"\nClass names saved to:"
)

print(class_file)