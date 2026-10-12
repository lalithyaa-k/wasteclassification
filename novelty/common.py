
import json
import re
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# PATHS  (ROOT = the folder that contains novelty/, src/, models/, ...)
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "processed" / "dataset"          # has train/ val/ test/
OUT = HERE / "results"                         # all new outputs go here
MODELS_OUT = HERE / "models"                   # new models (script 3) go here
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "resnet50": ROOT / "models" / "resnet50_waste.keras",
    "efficientnetb0": ROOT / "models" / "efficientnetb0_waste.keras",
}

# ----------------------------------------------------------------------------
# PREPROCESSING  -  *** THE MOST IMPORTANT SETTING TO CHECK ***
# Must match what your train_*.py / evaluate_*.py did.
#   "resnet50"     -> keras.applications.resnet50.preprocess_input (RGB->BGR, mean subtract)
#   "efficientnet" -> keras.applications.efficientnet.preprocess_input (does nothing; 0-255 raw,
#                     the EfficientNet model rescales internally)
#   "none"         -> raw 0-255 pixels
#   "scale01"      -> pixels / 255
#   "mobilenet"    -> pixels scaled to [-1, 1]
# ensemble.py prints each model's accuracy. Your earlier results were
# ResNet50 = 83.53% and EfficientNetB0 = 81.90%. If the printed numbers are far
# from these, change the mode here (open src/evaluate_*.py to see which one it used).
# ----------------------------------------------------------------------------
PREPROCESS = {
    "resnet50": "resnet50",
    "efficientnetb0": "efficientnet",
}

DEFAULT_IMG_SIZE = (224, 224)   # used only if the model's input size can't be read
BATCH = 32
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ----------------------------------------------------------------------------
# CLASS NAMES
# ----------------------------------------------------------------------------
def get_class_names():
    """Class order = results/class_names.json (what training saved).
    Falls back to the sorted folder names of processed/dataset/train."""
    folders = []
    train = DATA / "train"
    if train.exists():
        folders = sorted(p.name for p in train.iterdir() if p.is_dir())

    names = None
    jpath = ROOT / "results" / "class_names.json"
    if jpath.exists():
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = list(data.items())
            if all(isinstance(v, int) for _, v in items):      # {"Cardboard": 0, ...}
                names = [k for k, v in sorted(items, key=lambda kv: kv[1])]
            else:                                              # {"0": "Cardboard", ...}
                names = [v for k, v in sorted(items, key=lambda kv: int(kv[0]))]
        else:
            names = list(data)

    if names is None:
        if not folders:
            raise SystemExit("Could not find results/class_names.json or processed/dataset/train/*")
        names = folders
        print("[info] class_names.json not found; using sorted folder names:", names)

    if folders and [norm(n) for n in names] != [norm(f) for f in folders]:
        print("[WARNING] class_names.json order differs from sorted folder names!")
        print("   json   :", names)
        print("   folders:", folders)
        print("   Using the json order (that is what the model was trained with).")
    return names


CLASS_NAMES = get_class_names()
NUM_CLASSES = len(CLASS_NAMES)
_CLASS_LOOKUP = {norm(n): i for i, n in enumerate(CLASS_NAMES)}


# ----------------------------------------------------------------------------
# FILE COLLECTION
# ----------------------------------------------------------------------------
def collect_images(root):
    """Walk `root`; every image whose parent-folder path contains a class-name folder
    is returned as (Path, class_index). Works for train/<class>/x.jpg and also for
    <something>/train/<class>/x.jpg layouts."""
    root = Path(root)
    items = []
    if not root.exists():
        return items
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
            continue
        cls = None
        for part in reversed(p.relative_to(root).parts[:-1]):
            if norm(part) in _CLASS_LOOKUP:
                cls = _CLASS_LOOKUP[norm(part)]
                break
        if cls is not None:
            items.append((p, cls))
    return items


def collect_split(split):
    items = collect_images(DATA / split)
    if not items:
        raise SystemExit(f"No images found in {DATA / split}")
    return items


# ----------------------------------------------------------------------------
# TENSORFLOW HELPERS
# ----------------------------------------------------------------------------
def apply_preprocess(img, mode):
    import tensorflow as tf
    if mode == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(img)
    if mode == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input(img)
    if mode == "mobilenet":
        return tf.keras.applications.mobilenet_v2.preprocess_input(img)
    if mode == "scale01":
        return img / 255.0
    return img  # "none"


def make_ds(paths, mode, size, labels=None, batch=BATCH, shuffle=False, seed=0,
            skip_bad=False):
    import tensorflow as tf
    path_t = tf.constant([str(p) for p in paths])
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(path_t)
    else:
        ds = tf.data.Dataset.from_tensor_slices((path_t, np.asarray(labels, dtype=np.int32)))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 5000), seed=seed, reshuffle_each_iteration=True)

    def load(p):
        raw = tf.io.read_file(p)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, size)
        img = tf.cast(img, tf.float32)
        return apply_preprocess(img, mode)

    if labels is None:
        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda p, y: (load(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    if skip_bad:   # only safe for training (breaks alignment for prediction)
        ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def load_keras_model(path):
    import tensorflow as tf
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:  # e.g. Lambda layers need safe_mode=False
        print(f"[info] normal load failed ({type(e).__name__}); retrying with safe_mode=False "
              "(only do this for model files you trained yourself).")
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def get_all_probs(refresh=False, splits=("val", "test")):
    """Return softmax probabilities of both saved models on the given splits.
    Results are cached in novelty/results/probs_*.npy so inference runs only once."""
    labels = {s: np.array([c for _, c in collect_split(s)]) for s in splits}
    paths = {s: [p for p, _ in collect_split(s)] for s in splits}
    probs = {}
    for name, mpath in MODEL_PATHS.items():
        probs[name] = {}
        todo = []
        for s in splits:
            f = OUT / f"probs_{name}_{s}.npy"
            if f.exists() and not refresh:
                arr = np.load(f)
                if len(arr) == len(paths[s]) and arr.shape[1] == NUM_CLASSES:
                    probs[name][s] = arr
                    continue
            todo.append(s)
        if not todo:
            print(f"[cache] using saved predictions for {name}")
            continue

        print(f"[load] {mpath.name}")
        model = load_keras_model(mpath)
        out_units = model.output_shape[-1]
        if out_units != NUM_CLASSES:
            raise SystemExit(f"{name}: model has {out_units} outputs but there are "
                             f"{NUM_CLASSES} classes in class_names.json")
        shp = model.input_shape
        size = (int(shp[1]), int(shp[2])) if shp[1] and shp[2] else DEFAULT_IMG_SIZE
        print(f"       input size {size}, preprocessing '{PREPROCESS[name]}'")
        for s in todo:
            print(f"[predict] {name} on {s} ({len(paths[s])} images)")
            ds = make_ds(paths[s], PREPROCESS[name], size)
            p = model.predict(ds, verbose=1)
            if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):   # logits -> softmax
                e = np.exp(p - p.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
            np.save(OUT / f"probs_{name}_{s}.npy", p)
            probs[name][s] = p
    return {"probs": probs, "labels": labels, "paths": paths}


# ----------------------------------------------------------------------------
# METRICS / PLOTS
# ----------------------------------------------------------------------------
def summarize(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, title, out_path, normalize=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).astype(float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            txt = f"{v:.2f}" if normalize else f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if v > (0.5 if normalize else cm.max() / 2) else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)