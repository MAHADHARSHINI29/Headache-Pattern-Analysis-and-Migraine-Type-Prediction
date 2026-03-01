"""
Flask Web Application for Headache Pattern Analysis & Migraine Type Prediction
================================================================================
Serves a medical-themed symptom input form and returns AI-based predictions
with probability scores, risk assessment, and confidence explanations.

DISCLAIMER: This system is NOT for medical diagnosis.
            It is for early awareness and decision support only.
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from ml.utils import load_model, CLASS_LABELS
from ml.data_preprocessing import (
    NUMERIC_COLS, BINARY_COLS, CATEGORICAL_COLS,
    handle_missing_values, engineer_features,
)

# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Load model artifact once at startup
MODEL_ARTIFACT = None


def get_model():
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is None:
        MODEL_ARTIFACT = load_model()
    return MODEL_ARTIFACT


# ── Risk Scoring ─────────────────────────────────────────────────────────────
RISK_LEVELS = [
    (0.85, "High Confidence", "The model is highly confident in this prediction.", "#27ae60"),
    (0.60, "Moderate Confidence", "The model shows moderate certainty. Consider consulting a specialist.", "#f39c12"),
    (0.0,  "Low Confidence", "The model is uncertain. A professional evaluation is strongly recommended.", "#e74c3c"),
]


def get_risk_assessment(probabilities):
    """Return risk level based on max prediction probability."""
    max_prob = max(probabilities)
    for threshold, level, explanation, color in RISK_LEVELS:
        if max_prob >= threshold:
            return {
                "level": level,
                "explanation": explanation,
                "color": color,
                "confidence_pct": round(max_prob * 100, 1),
            }
    return RISK_LEVELS[-1]


# ── Feature Importance (for the result page) ─────────────────────────────────
def get_top_features(artifact, n=8):
    """Return top-n feature importances if available."""
    model = artifact["model"]
    names = artifact["feature_names"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        idx = np.argsort(imp)[-n:][::-1]
        return [{"name": names[i], "importance": round(imp[i] * 100, 1)} for i in idx]
    return []


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        artifact = get_model()
        model = artifact["model"]
        scaler = artifact["scaler"]
        label_encoders = artifact["label_encoders"]
        target_encoder = artifact["target_encoder"]
        feature_names = artifact["feature_names"]

        # ── Collect form data ────────────────────────────────────────────
        form = request.form.to_dict()

        row = {
            # Demographics
            "age": float(form.get("age", 30)),
            "gender": form.get("gender", "Male"),
            # Pain characteristics
            "pain_intensity": float(form.get("pain_intensity", 5)),
            "pain_location": form.get("pain_location", "Bilateral"),
            "pain_quality": form.get("pain_quality", "Pressing"),
            "duration_hours": float(form.get("duration_hours", 4)),
            # Associated symptoms
            "nausea": int(form.get("nausea", 0)),
            "vomiting": int(form.get("vomiting", 0)),
            "photophobia": int(form.get("photophobia", 0)),
            "phonophobia": int(form.get("phonophobia", 0)),
            "aura_present": int(form.get("aura_present", 0)),
            "aura_type": form.get("aura_type", "None"),
            "visual_disturbance": int(form.get("visual_disturbance", 0)),
            # Triggers / Lifestyle
            "stress_level": float(form.get("stress_level", 5)),
            "sleep_hours": float(form.get("sleep_hours", 7)),
            "physical_activity": form.get("physical_activity", "Moderate"),
            "caffeine_intake": float(form.get("caffeine_intake", 2)),
            "alcohol_intake": float(form.get("alcohol_intake", 1)),
            "weather_sensitivity": int(form.get("weather_sensitivity", 0)),
            "hormonal_factor": int(form.get("hormonal_factor", 0)),
            "screen_time": float(form.get("screen_time", 6)),
            # Clinical
            "frequency_per_month": int(form.get("frequency_per_month", 3)),
            "onset_pattern": form.get("onset_pattern", "Gradual"),
            "family_history": int(form.get("family_history", 0)),
            "medication_response": form.get("medication_response", "Moderate"),
        }

        # Build single-row DataFrame
        df = pd.DataFrame([row])

        # ── Preprocessing (same pipeline as training) ────────────────────
        # Add dummy target so the pipeline doesn't complain
        df["headache_type"] = "Migraine without aura"

        df = handle_missing_values(df)
        df = engineer_features(df)

        # Encode categoricals with saved encoders
        for col in CATEGORICAL_COLS:
            le = label_encoders[col]
            val = df[col].astype(str).values[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = 0  # fallback

        df.drop(columns=["headache_type"], inplace=True)

        # Ensure column order matches training
        df = df.reindex(columns=feature_names, fill_value=0)

        # Scale
        scale_cols = NUMERIC_COLS + [
            "severity_score", "frequency_index", "trigger_count", "symptom_count"
        ]
        scale_cols = [c for c in scale_cols if c in df.columns]
        df[scale_cols] = scaler.transform(df[scale_cols])

        # ── Predict ──────────────────────────────────────────────────────
        proba = model.predict_proba(df)[0]
        pred_idx = np.argmax(proba)
        pred_class = CLASS_LABELS[pred_idx]

        probabilities = [
            {"label": CLASS_LABELS[i], "prob": round(proba[i] * 100, 1)}
            for i in range(len(CLASS_LABELS))
        ]

        risk = get_risk_assessment(proba)
        top_features = get_top_features(artifact)

        return jsonify({
            "success": True,
            "prediction": pred_class,
            "probabilities": probabilities,
            "risk": risk,
            "top_features": top_features,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
