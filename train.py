from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from classifier import CLASSES_PATH, SCRIPT_DIR, X_PATH, Y_PATH
import numpy as np
import json
import joblib


MODEL_PATH = SCRIPT_DIR / "models" / "classifier.joblib"
MODEL_VERSION = "resnet18-logreg-v1"

def build_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

if __name__ == "__main__":
    # 1. Load cached features
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    classes = json.loads(CLASSES_PATH.read_text())
    print(f"X {X.shape}, y {y.shape}, classes: {classes}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = build_classifier()


    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_val)
    print(accuracy_score(y_val, pred))
    print(classification_report(y_val, pred, target_names=classes))
    print(confusion_matrix(y_val, pred))

    final_clf = build_classifier()
    final_clf.fit(X, y)
 
    # 5. Save model + what's needed to use it correctly
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"classifier": final_clf, "classes": classes, "model_version": MODEL_VERSION},
        MODEL_PATH,
    )
    print(f"\nSaved {MODEL_PATH} ({MODEL_VERSION})")