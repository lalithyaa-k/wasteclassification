

"""
reject_option.py  (Novelty 3)  -  "I'm not sure, send to a human".
 
Idea: if the highest probability (confidence) is below a threshold, the system
refuses to decide. Lower coverage (answers fewer images) should give higher
accuracy on the images it does answer.
 
The threshold is chosen on the VALIDATION set to reach a target accuracy, then
applied to the TEST set (so the test set is never used for tuning).
 
Works for ResNet50, EfficientNetB0 and the equal-weight ensemble.
Run ensemble.py first or let this script compute predictions itself (cached).
 
Run:   python reject_option.py
"""
import json
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
import common as C
 
TARGETS = [0.90, 0.95]          # accuracy we want on the answered images
MIN_ACCEPTED = 30               # ignore thresholds that answer fewer val images than this
 
 
def sweep(probs, y, thresholds):
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        acc = float((pred[keep] == y[keep]).mean()) if n else float("nan")
        rows.append({"threshold": float(t), "coverage": n / len(y), "accuracy_on_answered": acc,
                     "answered": n})
    return rows
 
 
def pick_threshold(val_rows, target):
    ok = [r for r in val_rows if r["answered"] >= MIN_ACCEPTED
          and r["accuracy_on_answered"] >= target]
    return min(ok, key=lambda r: r["threshold"])["threshold"] if ok else None
 
 
def main():
    d = C.get_all_probs()
    y_val, y_test = d["labels"]["val"], d["labels"]["test"]
    P = d["probs"]
    methods = {
        "ResNet50": (P["resnet50"]["val"], P["resnet50"]["test"]),
        "EfficientNetB0": (P["efficientnetb0"]["val"], P["efficientnetb0"]["test"]),
        "Ensemble": (0.5 * P["resnet50"]["val"] + 0.5 * P["efficientnetb0"]["val"],
                     0.5 * P["resnet50"]["test"] + 0.5 * P["efficientnetb0"]["test"]),
    }
    thresholds = np.linspace(0.30, 0.995, 140)
 
    results, fig_data = {}, {}
    print(f"\n{'method':15s}{'target':>8s}{'thr(val)':>10s}{'coverage':>10s}"
          f"{'acc answered':>14s}{'acc no-reject':>15s}")
    for name, (pv, pt) in methods.items():
        base_acc = float((pt.argmax(1) == y_test).mean())
        val_rows = sweep(pv, y_val, thresholds)
        test_rows = sweep(pt, y_test, thresholds)
        fig_data[name] = test_rows
        results[name] = {"accuracy_without_rejection": base_acc, "operating_points": []}
        for tgt in TARGETS:
            thr = pick_threshold(val_rows, tgt)
            if thr is None:
                print(f"{name:15s}{tgt*100:7.0f}%   target not reachable on validation set")
                results[name]["operating_points"].append({"target": tgt, "reachable": False})
                continue
            keep = pt.max(1) >= thr
            acc = float((pt.argmax(1)[keep] == y_test[keep]).mean()) if keep.any() else float("nan")
            cov = float(keep.mean())
            print(f"{name:15s}{tgt*100:7.0f}%{thr:10.3f}{cov*100:9.1f}%{acc*100:13.2f}%{base_acc*100:14.2f}%")
            # which classes get rejected most
            per_class = {}
            for c, cname in enumerate(C.CLASS_NAMES):
                m = y_test == c
                if m.any():
                    per_class[cname] = float(1 - keep[m].mean())
            results[name]["operating_points"].append({
                "target": tgt, "reachable": True, "threshold_from_val": thr,
                "test_coverage": cov, "test_accuracy_on_answered": acc,
                "rejected_images": int((~keep).sum()),
                "rejection_rate_per_class": per_class})
        results[name]["sweep_test"] = test_rows
 
    # ---- plot: accuracy vs coverage ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in fig_data.items():
        cov = [r["coverage"] for r in rows]
        acc = [r["accuracy_on_answered"] for r in rows]
        ax.plot(cov, acc, label=name)
    ax.set_xlabel("Coverage (fraction of test images the system answers)")
    ax.set_ylabel("Accuracy on answered images")
    ax.set_title("Reject option: accuracy vs coverage (test set)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(C.OUT / "reject_option_accuracy_vs_coverage.png", dpi=150)
    plt.close(fig)
 
    with open(C.OUT / "reject_option_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", C.OUT)
    print("Tip for your report: the curve going UP as coverage goes DOWN means the "
          "model's confidence is meaningful.")
 
 
if __name__ == "__main__":
    main()
 

