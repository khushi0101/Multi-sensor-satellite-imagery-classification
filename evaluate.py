import joblib
import csv
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from classifier import BASE_DIR, SCRIPT_DIR, extract_features, load_feature_extractor

MODEL_PATH = SCRIPT_DIR / "models" / "classifier.joblib"
LABELS_CSV = BASE_DIR / "eval_labels.csv"
EVAL_DIR = BASE_DIR / "eval_set"
PREDICTIONS_CSV = SCRIPT_DIR / "eval_predictions.csv"
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def load_eval_labels():
    """eval_labels.csv -> list of (filename, true_label). strip() handles Windows line endings."""
    rows = []
    with open(LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["filename"].strip(), row["true_label"].strip()))
    return rows


if __name__ == "__main__":
    bundle = joblib.load(MODEL_PATH)
    clf, classes, version = bundle["classifier"], bundle["classes"], bundle["model_version"]
    net = load_feature_extractor()
    print(f"Model: {version}")

    rows = load_eval_labels()
    paths = [EVAL_DIR / fname for fname, _ in rows]
    y_true = np.array([classes.index(label) for _, label in rows])
    print(f"{len(paths)} eval tiles")


    X = extract_features(net, paths)
    probs = clf.predict_proba(X)
    y_pred = probs.argmax(axis=1)           # index of highest probability
    confidence = probs.max(axis=1)          # that highest probability
    correct = y_pred == y_true

    # 4. Standard metrics
    print(f"\nEval accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred))
 
    # 5. Are wrong predictions less confident?
    print(f"\nMean confidence when correct: {confidence[correct].mean():.3f}")
    if (~correct).any():
        print(f"Mean confidence when wrong:   {confidence[~correct].mean():.3f}")
 
    # 6. Threshold table: coverage vs accuracy
    print("\nThreshold | Coverage (auto-accepted) | Accuracy on accepted | Sent to review")
    for t in THRESHOLDS:
        accepted = confidence >= t
        coverage = accepted.mean()
        acc = correct[accepted].mean() if accepted.any() else float("nan")
        print(f"  {t:>5.2f}   | {coverage:>6.1%} ({accepted.sum():>3})         "
              f"| {acc:>6.1%}               | {(~accepted).sum():>3}")
 
    # 7. Save per-tile results for inspection
    with open(PREDICTIONS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "predicted_label", "confidence", "correct"])
        for (fname, true_label), p, c, ok in zip(rows, y_pred, confidence, correct):
            w.writerow([fname, true_label, classes[p], f"{c:.3f}", ok])
    print(f"\nPer-tile results saved to {PREDICTIONS_CSV}")