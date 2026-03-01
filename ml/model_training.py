"""
Model Training for Headache Pattern Analysis
==============================================
Trains Random Forest, SVM, and Gradient Boosting classifiers.
Performs hyperparameter tuning and saves the best model.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.data_preprocessing import preprocess
from ml.utils import save_model, save_model_comparison, save_feature_importance


# ── Hyperparameter Grids ────────────────────────────────────────────────────
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
}


# ── Model Definitions ───────────────────────────────────────────────────────
def _get_base_models():
    return {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


# ── Training Loop ───────────────────────────────────────────────────────────
def train_and_tune(X_train, y_train, X_test, y_test):
    """
    Train each model with GridSearchCV, evaluate on test set.

    Returns
    -------
    results      : dict  {name: {accuracy, precision, recall, f1, model, time}}
    best_model   : fitted estimator with highest F1 score
    best_name    : name of the best model
    """
    models = _get_base_models()
    results = {}

    print("── Model Training & Hyperparameter Tuning ─────────")
    for name, model in models.items():
        print(f"\n   🔧 {name}")
        t0 = time.time()

        grid = GridSearchCV(
            model,
            PARAM_GRIDS[name],
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        y_pred = best.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        elapsed = time.time() - t0

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "model": best,
            "best_params": grid.best_params_,
            "time": round(elapsed, 1),
        }

        print(f"      Best params : {grid.best_params_}")
        print(f"      Accuracy    : {acc:.4f}")
        print(f"      Precision   : {prec:.4f}")
        print(f"      Recall      : {rec:.4f}")
        print(f"      F1-score    : {f1:.4f}")
        print(f"      Time        : {elapsed:.1f}s")

    # Pick the best model by F1 score
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\n   🏆 Best Model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

    return results, best_model, best_name


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Preprocess
    X_train, X_test, y_train, y_test, artifacts = preprocess()

    # 2. Train & tune
    results, best_model, best_name = train_and_tune(X_train, y_train, X_test, y_test)

    # 3. Save comparison chart
    chart_data = {k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1"]}
                  for k, v in results.items()}
    save_model_comparison(chart_data)

    # 4. Feature importance (tree-based models)
    if hasattr(best_model, "feature_importances_"):
        save_feature_importance(
            best_model.feature_importances_,
            artifacts["feature_names"],
        )

    # 5. Save the best model + artifacts
    model_artifact = {
        "model": best_model,
        "model_name": best_name,
        "scaler": artifacts["scaler"],
        "label_encoders": artifacts["label_encoders"],
        "target_encoder": artifacts["target_encoder"],
        "feature_names": artifacts["feature_names"],
        "results": {k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1"]}
                    for k, v in results.items()},
    }
    save_model(model_artifact)

    return results, best_model, best_name


if __name__ == "__main__":
    main()
