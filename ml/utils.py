"""
Utility Functions for Headache Pattern Analysis Project
========================================================
Shared helpers for data loading, model persistence, and plotting.
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PLOT_DIR = os.path.join(PROJECT_ROOT, "static", "plots")

DATASET_PATH = os.path.join(DATA_DIR, "headache_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

CLASS_LABELS = [
    "Migraine without aura",
    "Migraine with aura",
    "Tension-type headache",
    "Cluster headache",
]

# Ensure directories exist
for d in [DATA_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Data I/O ─────────────────────────────────────────────────────────────────
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the headache dataset CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python ml/generate_dataset.py` first."
        )
    return pd.read_csv(path)


# ── Model I/O ────────────────────────────────────────────────────────────────
def save_model(artifact: dict, path: str = MODEL_PATH):
    """
    Save model artifact (dict with model, scaler, encoders, feature names, etc.)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(artifact, path)
    print(f"✅ Model artifact saved → {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    """Load model artifact."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run `python ml/model_training.py` first."
        )
    return joblib.load(path)


# ── Plotting Helpers ─────────────────────────────────────────────────────────
def save_confusion_matrix(cm, labels, title="Confusion Matrix", filename="confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   📊 Saved → {path}")


def save_feature_importance(importances, feature_names, top_n=15,
                            filename="feature_importance.png"):
    """Plot and save a horizontal bar chart of feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(idx)), importances[idx], color="#4f8cf7")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   📊 Saved → {path}")


def save_model_comparison(results: dict, filename="model_comparison.png"):
    """
    Bar chart comparing models on Accuracy, Precision, Recall, F1.

    Parameters
    ----------
    results : dict
        {model_name: {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}}
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4f8cf7", "#f74f4f", "#4fc74f"]
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   📊 Saved → {path}")
