
from pathlib import Path
import json
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight


# PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "processed" / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_waste.keras"
RESULTS_PATH = PROJECT_ROOT / "results"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# SETTINGS

IMAGE_SIZE = (224, 224)
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

tf.keras.utils.set_random_seed(SEED)


# LOAD DATASETS

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


# RESNET50 PREPROCESSING

def preprocess_dataset(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels


train_dataset = train_dataset.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)


# CLASS WEIGHTS

print("\nCalculating class weights...")

class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for _, labels in train_dataset:
    class_counts += np.bincount(
        labels.numpy(),
        minlength=NUM_CLASSES
    )

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.repeat(np.arange(NUM_CLASSES), class_counts)
)

class_weights = {
    i: float(class_weights_array[i])
    for i in range(NUM_CLASSES)
}

print("\nClass distribution:")

for i, name in enumerate(CLASSES):
    print(
        f"{name:25} : {class_counts[i]:6d} "
        f"| weight = {class_weights[i]:.4f}"
    )


# BUILD RESNET50

print("\nBuilding ResNet50...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)


# PHASE 1: FEATURE EXTRACTION

print("\nPHASE 1: FEATURE EXTRACTION")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase1 = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

start_time = time.time()

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks_phase1
)


# PHASE 2: FINE-TUNING

print("\nPHASE 2: FINE-TUNING")

base_model.trainable = True

fine_tune_at = int(len(base_model.layers) * 0.7)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Freeze Batch Normalization layers for stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks_phase2
)


# SAVE MODEL

model.save(MODEL_PATH)

class_file = RESULTS_PATH / "class_names.json"

with open(class_file, "w", encoding="utf-8") as file:
    json.dump(CLASSES, file, indent=4)


# FINAL VALIDATION

loss, accuracy = model.evaluate(val_dataset, verbose=1)
elapsed_time = time.time() - start_time

summary = {
    "model": "ResNet50",
    "validation_loss": float(loss),
    "validation_accuracy": float(accuracy),
    "training_time_seconds": round(elapsed_time, 2),
    "class_names": CLASSES
}

summary_path = RESULTS_PATH / "resnet50_results.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nTRAINING COMPLETE")
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Training time: {elapsed_time / 60:.2f} minutes")
print(f"Model saved to: {MODEL_PATH}")
print(f"Results saved to: {summary_path}")
