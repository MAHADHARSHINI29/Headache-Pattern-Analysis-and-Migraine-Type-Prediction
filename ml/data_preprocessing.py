"""
Data Preprocessing for Headache Pattern Analysis
==================================================
Handles: missing values, encoding, feature engineering, scaling, train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.utils import load_dataset, CLASS_LABELS


# ── Feature Definitions ─────────────────────────────────────────────────────
NUMERIC_COLS = [
    "age", "pain_intensity", "duration_hours",
    "stress_level", "sleep_hours", "caffeine_intake",
    "alcohol_intake", "screen_time", "frequency_per_month",
]

BINARY_COLS = [
    "nausea", "vomiting", "photophobia", "phonophobia",
    "aura_present", "visual_disturbance",
    "weather_sensitivity", "hormonal_factor", "family_history",
]

CATEGORICAL_COLS = [
    "gender", "pain_location", "pain_quality",
    "aura_type", "physical_activity",
    "onset_pattern", "medication_response",
]

TARGET_COL = "headache_type"


# ── Missing-Value Handling ───────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode for categorical/binary."""
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            df[col] = df[col].astype(int)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # Safety net: fill any remaining NaN values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Unknown")

    return df


# ── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that may improve classification."""
    df = df.copy()

    # Severity Score: weighted combination of pain intensity, nausea, vomiting
    df["severity_score"] = (
        df["pain_intensity"] * 0.5
        + df["nausea"].astype(float) * 2.0
        + df["vomiting"].astype(float) * 2.5
        + df["photophobia"].astype(float) * 1.5
        + df["phonophobia"].astype(float) * 1.5
    )

    # Frequency Index: normalized frequency × duration
    max_freq = df["frequency_per_month"].max() if df["frequency_per_month"].max() > 0 else 1
    max_dur = df["duration_hours"].max() if df["duration_hours"].max() > 0 else 1
    df["frequency_index"] = (
        (df["frequency_per_month"] / max_freq) * 0.6
        + (df["duration_hours"] / max_dur) * 0.4
    )

    # Trigger Count: how many lifestyle triggers are elevated
    df["trigger_count"] = (
        (df["stress_level"] > 6).astype(int)
        + (df["sleep_hours"] < 6).astype(int)
        + (df["caffeine_intake"] > 4).astype(int)
        + (df["alcohol_intake"] > 3).astype(int)
        + df["weather_sensitivity"].astype(int)
        + df["hormonal_factor"].astype(int)
        + (df["screen_time"] > 8).astype(int)
    )

    # Symptom Count: total associated symptoms present
    df["symptom_count"] = (
        df["nausea"].astype(int)
        + df["vomiting"].astype(int)
        + df["photophobia"].astype(int)
        + df["phonophobia"].astype(int)
        + df["aura_present"].astype(int)
        + df["visual_disturbance"].astype(int)
    )

    return df


# ── Encoding ─────────────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    """
    Label-encode categorical features; encode target column.

    Returns
    -------
    df : pd.DataFrame
        Encoded DataFrame (target removed).
    y : np.ndarray
        Encoded target array.
    label_encoders : dict
        {column_name: fitted LabelEncoder}.
    target_encoder : LabelEncoder
        Fitted encoder for headache_type.
    """
    df = df.copy()
    label_encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(CLASS_LABELS)
    y = target_encoder.transform(df[TARGET_COL])
    df.drop(columns=[TARGET_COL], inplace=True)

    return df, y, label_encoders, target_encoder


# ── Scaling ──────────────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame, scaler=None):
    """
    StandardScaler on numeric + engineered columns.

    Returns
    -------
    df_scaled : pd.DataFrame
    scaler    : fitted StandardScaler
    """
    scale_cols = NUMERIC_COLS + [
        "severity_score", "frequency_index", "trigger_count", "symptom_count"
    ]
    scale_cols = [c for c in scale_cols if c in df.columns]

    if scaler is None:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df, scaler


# ── Full Pipeline ────────────────────────────────────────────────────────────
def preprocess(test_size=0.2, random_state=42):
    """
    Run the complete preprocessing pipeline.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : np.ndarray
    artifacts       : dict  (scaler, label_encoders, target_encoder, feature_names)
    """
    print("── Data Preprocessing ──────────────────────────────")
    df = load_dataset()
    print(f"   Raw shape       : {df.shape}")

    # 1. Missing values
    df = handle_missing_values(df)
    print(f"   After imputation : {df.isnull().sum().sum()} missing values remain")

    # 2. Feature engineering
    df = engineer_features(df)
    print(f"   Engineered cols  : severity_score, frequency_index, trigger_count, symptom_count")

    # 3. Encoding
    df, y, label_encoders, target_encoder = encode_features(df)
    print(f"   Encoded features : {len(CATEGORICAL_COLS)} categorical columns")

    # 4. Train / Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train / Test     : {X_train.shape[0]} / {X_test.shape[0]}")

    # 5. Scaling (fit on train only)
    X_train, scaler = scale_features(X_train.copy())
    X_test, _       = scale_features(X_test.copy(), scaler=scaler)

    feature_names = list(df.columns)

    artifacts = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "feature_names": feature_names,
    }

    print("   ✅ Preprocessing complete\n")
    return X_train, X_test, y_train, y_test, artifacts


# ── Standalone Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, artifacts = preprocess()
    print(f"Feature count: {len(artifacts['feature_names'])}")
    print(f"Features: {artifacts['feature_names']}")
