"""
ensemble.py  (Novelty 1)  -  combine ResNet50 + EfficientNetB0.

Soft voting: average the two models' probabilities for every image.
  * equal weights (0.5 / 0.5)
  * weight tuned on the VALIDATION set (never on test), then reported on test

No training. Reads the saved .keras models, writes only to novelty/results/.

Run:   python ensemble.py            (add --refresh to recompute predictions)
"""
import argparse
import json

import numpy as np
from sklearn.metrics import classification_report

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore cached predictions")
    args = ap.parse_args()

    d = C.get_all_probs(refresh=args.refresh)
    y_val, y_test = d["labels"]["val"], d["labels"]["test"]
    r, e = d["probs"]["resnet50"], d["probs"]["efficientnetb0"]

    # ---- individual models -------------------------------------------------
    res = {}
    for name in ("resnet50", "efficientnetb0"):
        res[name] = C.summarize(y_test, d["probs"][name]["test"].argmax(1))
    print("\nSanity check - your earlier reports: ResNet50 83.53%, EfficientNetB0 81.90%")
    for name in res:
        print(f"  {name:15s} test accuracy now: {res[name]['accuracy']*100:.2f}%")
    print("  (if these are far off, fix PREPROCESS in common.py before trusting anything below)\n")

    # ---- equal-weight ensemble --------------------------------------------
    p_eq = 0.5 * r["test"] + 0.5 * e["test"]
    res["ensemble_equal"] = C.summarize(y_test, p_eq.argmax(1))

    # ---- weight tuned on validation ---------------------------------------
    best_w, best_f1 = 0.5, -1
    grid = {}
    for w in np.linspace(0, 1, 11):
        pv = w * r["val"] + (1 - w) * e["val"]
        f1 = C.summarize(y_val, pv.argmax(1))["macro_f1"]
        grid[f"{w:.1f}"] = f1
        if f1 > best_f1:
            best_w, best_f1 = float(w), f1
    p_tuned = best_w * r["test"] + (1 - best_w) * e["test"]
    res["ensemble_tuned"] = C.summarize(y_test, p_tuned.argmax(1))
    res["ensemble_tuned"]["resnet_weight_chosen_on_val"] = best_w

    # ---- do the models complement each other? -----------------------------
    pr, pe = r["test"].argmax(1), e["test"].argmax(1)
    cr, ce = pr == y_test, pe == y_test
    comp = {
        "both_correct": float(np.mean(cr & ce)),
        "only_resnet_correct": float(np.mean(cr & ~ce)),
        "only_efficientnet_correct": float(np.mean(~cr & ce)),
        "both_wrong": float(np.mean(~cr & ~ce)),
        "oracle_upper_bound_accuracy": float(np.mean(cr | ce)),
    }

    # ---- print table ------------------------------------------------------
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    print(f"{'model':18s}" + "".join(f"{c:>20s}" for c in cols))
    for k in ("resnet50", "efficientnetb0", "ensemble_equal", "ensemble_tuned"):
        print(f"{k:18s}" + "".join(f"{res[k][c]*100:19.2f}%" for c in cols))
    print(f"\nTuned ResNet weight (picked on val): {best_w:.1f}")
    print("Complementarity on test:", json.dumps({k: round(v, 4) for k, v in comp.items()}, indent=2))

    # ---- save -------------------------------------------------------------
    with open(C.OUT / "ensemble_results.json", "w") as f:
        json.dump({"test_metrics": res, "val_macro_f1_by_resnet_weight": grid,
                   "complementarity_test": comp, "class_names": C.CLASS_NAMES}, f, indent=2)
    with open(C.OUT / "ensemble_classification_report.txt", "w") as f:
        f.write("ENSEMBLE (equal weights) - test set\n\n")
        f.write(classification_report(y_test, p_eq.argmax(1), labels=list(range(C.NUM_CLASSES)),
                                      target_names=C.CLASS_NAMES, digits=4, zero_division=0))
    C.save_confusion_matrix(y_test, p_eq.argmax(1), "Ensemble (equal weights) - normalized",
                            C.OUT / "ensemble_confusion_matrix.png")
    print("\nSaved to", C.OUT)


if __name__ == "__main__":
    main()