"""
leave_one_dataset_out.py  (Novelty 8)  -  train on two datasets, test on the third.

Question answered: does the model understand WASTE, or did it just learn the
"look" of each dataset (camera, background, lighting)?

It trains a NEW model (your saved models are never touched or overwritten) with
a fixed, simple recipe (frozen pretrained base -> short fine-tune), so the
comparison between conditions is fair:

    --holdout rcd         train on REALWASTE + TACO, test on RCD
    --holdout realwaste   train on TACO + RCD, test on REALWASTE
    --holdout taco        train on REALWASTE + RCD, test on TACO (only ~131 crops -> noisy)
    --holdout none        BASELINE: same pooled data, ordinary random split
                          (compare this to the runs above = the "domain gap")

Run order suggestion (same --model and --seed for all):
    python leave_one_dataset_out.py --inspect
    python leave_one_dataset_out.py --holdout none
    python leave_one_dataset_out.py --holdout rcd
    python leave_one_dataset_out.py --holdout realwaste

HOW THE SCRIPT KNOWS WHICH DATASET AN IMAGE CAME FROM
  --mode processed (default): looks at processed/dataset/*/<class>/<file>. It guesses the
        source from the file/folder name (keywords below). Use --inspect to see if that works.
  --mode dirs: if your processed filenames do NOT reveal the source, edit SOURCE_DIRS below to
        point at your three prepared folders (each must contain <class name>/ subfolders).

Time: this is real training. On CPU use --max-per-class 300 --epochs 3 --ft-epochs 2 first.
Needs internet the first time unless ImageNet weights are already cached by Keras.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import common as C


SOURCE_KEYWORDS = {
    "realwaste": ["realwaste", "real_waste", "real-waste"],
    "taco": ["taco"],
    "rcd": ["rcd", "remondis"],
}

IMG_SIZE = (224, 224)

PRE = {
    "resnet50": "resnet50",
    "efficientnetb0": "efficientnet",
}


# --------------------------------------------------
# 1. DATASET SOURCE DETECTION
# --------------------------------------------------

def detect_source(path):
    path = Path(path)

    try:
        text = str(path.relative_to(C.DATA)).lower()
    except ValueError:
        text = str(path).lower()

    for source, keywords in SOURCE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return source

    return "unknown"


def gather():
    rows = []

    for path, class_index in C.collect_images(C.DATA):
        rows.append((Path(path), int(class_index), detect_source(path)))

    return rows


def show_table(rows):
    sources = sorted(set(row[2] for row in rows))

    print(
        f"\n{'Class':22s}"
        + "".join(f"{source:>12s}" for source in sources)
        + f"{'Total':>10s}"
    )

    for class_index, class_name in enumerate(C.CLASS_NAMES):
        counts = [
            sum(
                1 for row in rows
                if row[1] == class_index and row[2] == source
            )
            for source in sources
        ]

        print(
            f"{class_name:22s}"
            + "".join(f"{count:12d}" for count in counts)
            + f"{sum(counts):10d}"
        )

    totals = [
        sum(1 for row in rows if row[2] == source)
        for source in sources
    ]

    print(
        f"{'TOTAL':22s}"
        + "".join(f"{total:12d}" for total in totals)
        + f"{sum(totals):10d}\n"
    )


def inspect(rows):
    print(f"Found {len(rows)} labelled images under {C.DATA}")
    show_table(rows)

    for source in sorted(set(row[2] for row in rows)):
        examples = [row[0] for row in rows if row[2] == source]

        print(f"\nExamples for {source}:")
        for path in examples[:5]:
            print(" ", path)


# --------------------------------------------------
# 2. DATASET SPLITTING
# --------------------------------------------------

def strat_split(indices, labels, test_size, seed):
    indices = np.asarray(indices)
    labels = np.asarray(labels)

    if len(indices) < 2:
        raise SystemExit("Not enough images to split.")

    _, counts = np.unique(labels, return_counts=True)

    stratify = (
        labels
        if len(counts) > 0 and counts.min() >= 2
        else None
    )

    return train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


# --------------------------------------------------
# 3. IMAGE DATASET CREATION
# --------------------------------------------------

def check_images(paths, labels):
    valid_paths = []
    valid_labels = []
    skipped = 0

    for path, label in zip(paths, labels):
        try:
            raw = tf.io.read_file(str(path))

            image = tf.io.decode_image(
                raw,
                channels=3,
                expand_animations=False,
            )

            # Force decoding now, before training starts.
            image.numpy()

            valid_paths.append(str(path))
            valid_labels.append(int(label))

        except Exception as error:
            skipped += 1
            print(f"Skipping unreadable image: {path}")

    if skipped:
        print(f"Skipped {skipped} unreadable image(s).")

    if not valid_paths:
        raise SystemExit("No readable images remain.")

    return valid_paths, np.asarray(valid_labels, dtype=np.int32)


def make_dataset(
    paths,
    labels,
    preprocess,
    batch_size,
    shuffle=False,
    seed=42,
):
    paths, labels = check_images(paths, labels)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        raw = tf.io.read_file(path)

        image = tf.io.decode_image(
            raw,
            channels=3,
            expand_animations=False,
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)

        image = C.apply_preprocess(image, preprocess)

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Calculate the number of batches from validated image count.
    steps = (len(paths) + batch_size - 1) // batch_size

    return dataset, steps, len(paths)


def make_test_dataset(paths, preprocess, batch_size):
    paths = [str(path) for path in paths]

    dataset = tf.data.Dataset.from_tensor_slices(paths)

    def load_image(path):
        raw = tf.io.read_file(path)

        image = tf.io.decode_image(
            raw,
            channels=3,
            expand_animations=False,
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)

        return C.apply_preprocess(image, preprocess)

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# 4. MODEL CREATION
# --------------------------------------------------

def build_model(model_name, num_classes):
    from tensorflow.keras import layers

    model_args = {
        "include_top": False,
        "weights": "imagenet",
        "input_shape": IMG_SIZE + (3,),
    }

    if model_name == "resnet50":
        base = tf.keras.applications.ResNet50(**model_args)
    else:
        base = tf.keras.applications.EfficientNetB0(**model_args)

    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    x = augmentation(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base


# --------------------------------------------------
# 5. MAIN EXPERIMENT
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-dataset-out generalization experiment"
    )

    parser.add_argument(
        "--holdout",
        choices=["none", "rcd", "realwaste", "taco"],
        default="rcd",
    )

    parser.add_argument(
        "--model",
        choices=["resnet50", "efficientnetb0"],
        default="resnet50",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ft-epochs", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")

    if args.ft_epochs < 0 or args.batch_size < 1:
        raise SystemExit("Invalid fine-tuning epochs or batch size.")

    if args.max_per_class < 0:
        raise SystemExit("--max-per-class cannot be negative.")

    rows = gather()

    if not rows:
        raise SystemExit("No labelled images found.")

    if args.inspect:
        inspect(rows)
        return

    unknown = [row for row in rows if row[2] == "unknown"]

    if unknown:
        inspect(rows)
        raise SystemExit(
            "Some images have unknown sources. Fix source detection before training."
        )

    show_table(rows)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    indices = np.arange(len(rows))

    labels = np.asarray(
        [row[1] for row in rows],
        dtype=np.int32,
    )

    sources = np.asarray(
        [row[2] for row in rows]
    )

    # Baseline: random test split from the combined dataset.
    if args.holdout == "none":
        pool_indices, test_indices = strat_split(
            indices,
            labels,
            test_size=0.15,
            seed=args.seed,
        )

    # Cross-dataset evaluation: entire source held out for testing.
    else:
        test_indices = indices[sources == args.holdout]
        pool_indices = indices[sources != args.holdout]

        if len(test_indices) == 0:
            raise SystemExit(
                f"No images found for holdout source: {args.holdout}"
            )

        if len(pool_indices) < 2:
            raise SystemExit("Not enough images for training and validation.")

    train_indices, val_indices = strat_split(
        pool_indices,
        labels[pool_indices],
        test_size=0.15,
        seed=args.seed,
    )

    # Limit training images per class to reduce CPU workload.
    if args.max_per_class > 0:
        rng = np.random.RandomState(args.seed)
        selected = []

        for class_index in range(C.NUM_CLASSES):
            class_indices = train_indices[
                labels[train_indices] == class_index
            ].copy()

            rng.shuffle(class_indices)

            selected.extend(
                class_indices[:args.max_per_class].tolist()
            )

        train_indices = np.asarray(
            sorted(selected),
            dtype=np.int64,
        )

    print(
        f"Train: {len(train_indices)} | "
        f"Validation: {len(val_indices)} | "
        f"Test: {len(test_indices)}"
    )

    # Identify test classes absent from the training data.
    train_classes = set(labels[train_indices].tolist())
    test_classes = set(labels[test_indices].tolist())

    missing_classes = [
        C.CLASS_NAMES[index]
        for index in sorted(test_classes - train_classes)
    ]

    if missing_classes:
        print(
            "\nWARNING: Test classes absent from training:",
            missing_classes,
        )

    if args.holdout == "rcd":
        print(
            "\nNOTE: Your earlier inspection found RCD images only "
            "in the Plastic class. Verify the RCD label mapping before "
            "interpreting this as nine-class generalization."
        )

    preprocess = PRE[args.model]

    train_ds, train_steps, train_count = make_dataset(
        [rows[i][0] for i in train_indices],
        labels[train_indices],
        preprocess,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    val_ds, val_steps, val_count = make_dataset(
        [rows[i][0] for i in val_indices],
        labels[val_indices],
        preprocess,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    print(f"Readable training images: {train_count}")
    print(f"Readable validation images: {val_count}")
    print(f"Training batches per epoch: {train_steps}")
    print(f"Validation batches: {val_steps}")

    class_weight = None

    if args.class_weights:
        present_classes = np.asarray(
            sorted(set(labels[train_indices].tolist()))
        )

        weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=labels[train_indices],
        )

        class_weight = {
            int(class_id): float(weight)
            for class_id, weight in zip(present_classes, weights)
        }

    model, base = build_model(
        args.model,
        C.NUM_CLASSES,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    if args.ft_epochs > 0:
        base.trainable = True

        for layer in base.layers[:-30]:
            layer.trainable = False

        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.ft_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    train_seconds = time.time() - start_time

    # --------------------------------------------------
    # 6. TEST EVALUATION
    # --------------------------------------------------

    test_paths = [rows[i][0] for i in test_indices]
    y_true = labels[test_indices]

    test_ds = make_test_dataset(
        test_paths,
        preprocess,
        args.batch_size,
    )

    probabilities = model.predict(test_ds, verbose=1)

    if len(probabilities) != len(y_true):
        raise SystemExit(
            f"Prediction count mismatch: {len(probabilities)} predictions "
            f"for {len(y_true)} labels."
        )

    y_pred = np.argmax(probabilities, axis=1)

    present_classes = sorted(set(y_true.tolist()))

    report = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=[
            C.CLASS_NAMES[index]
            for index in present_classes
        ],
        digits=4,
        zero_division=0,
    )

    metrics = C.summarize(y_true, y_pred)

    print("\n========== TEST RESULTS ==========")
    print(f"Holdout: {args.holdout}")
    print(f"Model: {args.model}")
    print(f"Test images: {len(y_true)}")
    print(report)
    print("Summary metrics:", metrics)

    # --------------------------------------------------
    # 7. SAVE RESULTS
    # --------------------------------------------------

    tag = f"{args.model}_holdout_{args.holdout}"

    C.OUT.mkdir(parents=True, exist_ok=True)
    C.MODELS_OUT.mkdir(parents=True, exist_ok=True)

    model.save(C.MODELS_OUT / f"{tag}.keras")

    C.save_confusion_matrix(
        y_true,
        y_pred,
        f"{args.model}: holdout={args.holdout}",
        C.OUT / f"loo_{tag}_confusion_matrix.png",
    )

    with open(
        C.OUT / f"loo_{tag}_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    result = {
        "holdout": args.holdout,
        "model": args.model,
        "seed": args.seed,
        "sizes": {
            "train": int(len(train_indices)),
            "validation": int(len(val_indices)),
            "test": int(len(test_indices)),
            "readable_train": int(train_count),
            "readable_validation": int(val_count),
        },
        "test_classes_present": [
            C.CLASS_NAMES[index]
            for index in present_classes
        ],
        "test_classes_missing_in_training": missing_classes,
        "metrics_on_test": {
            key: float(value)
            for key, value in metrics.items()
        },
        "train_seconds": float(train_seconds),
        "settings": vars(args),
    }

    with open(
        C.OUT / f"loo_{tag}_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    print(f"\nResults saved to: {C.OUT}")
    print(f"Experiment model saved to: {C.MODELS_OUT}")


if __name__ == "__main__":
    main()