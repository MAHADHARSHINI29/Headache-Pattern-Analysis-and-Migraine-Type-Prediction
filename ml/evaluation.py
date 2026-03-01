"""
Model Evaluation for Headache Pattern Analysis
================================================
Generates confusion matrices, classification reports, and summary visualizations.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ml.data_preprocessing import preprocess
from ml.utils import (
    load_model,
    save_confusion_matrix,
    save_feature_importance,
    save_model_comparison,
    CLASS_LABELS,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Print full classification report and save confusion matrix.

    Returns
    -------
    metrics : dict  {accuracy, precision, recall, f1}
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n── Evaluation: {model_name} ────────────────────────")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(
        cm,
        labels=CLASS_LABELS,
        title=f"Confusion Matrix — {model_name}",
        filename=f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def run_full_evaluation():
    """Load saved model, run preprocessing, evaluate, and regenerate all plots."""
    # 1. Load model artifact
    artifact = load_model()
    model = artifact["model"]
    model_name = artifact["model_name"]
    feature_names = artifact["feature_names"]

    # 2. Preprocess data (same pipeline)
    X_train, X_test, y_train, y_test, _ = preprocess()

    # 3. Evaluate
    metrics = evaluate_model(model, X_test, y_test, model_name)

    # 4. Feature importance
    if hasattr(model, "feature_importances_"):
        save_feature_importance(model.feature_importances_, feature_names)

    # 5. Model comparison chart (from saved results)
    if "results" in artifact:
        save_model_comparison(artifact["results"])

    print("\n   ✅ Full evaluation complete")
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_full_evaluation()
