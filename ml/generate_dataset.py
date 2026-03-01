"""
Synthetic Dataset Generator for Headache Pattern Analysis & Migraine Type Prediction
=====================================================================================
Generates a clinically-informed synthetic dataset with realistic probability
distributions for 4 headache types:
  1. Migraine without aura
  2. Migraine with aura
  3. Tension-type headache
  4. Cluster headache

Author : Final Year B.Tech AI Project
Python : 3.10+
"""

import os
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Configuration ────────────────────────────────────────────────────────────
N_SAMPLES = 5000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "headache_dataset.csv")

HEADACHE_TYPES = [
    "Migraine without aura",
    "Migraine with aura",
    "Tension-type headache",
    "Cluster headache",
]

# Class distribution (prevalence-inspired)
CLASS_WEIGHTS = [0.35, 0.20, 0.35, 0.10]


# ── Helper Functions ─────────────────────────────────────────────────────────
def _clamp(arr, lo, hi):
    """Clamp array values to [lo, hi]."""
    return np.clip(arr, lo, hi)


def _pick(options, probs, n):
    """Randomly pick from options with given probabilities."""
    return np.random.choice(options, size=n, p=probs)


# ── Per-Class Generators ────────────────────────────────────────────────────
def _generate_migraine_without_aura(n: int) -> pd.DataFrame:
    """Generate rows for Migraine without aura."""
    data = {
        "age": _clamp(np.random.normal(35, 12, n).astype(int), 12, 75),
        "gender": _pick(["Male", "Female", "Other"], [0.30, 0.65, 0.05], n),
        "pain_intensity": _clamp(np.random.normal(7.5, 1.2, n).round(1), 1, 10),
        "pain_location": _pick(
            ["Unilateral", "Bilateral", "Frontal", "Temporal", "Occipital"],
            [0.55, 0.15, 0.10, 0.15, 0.05], n,
        ),
        "pain_quality": _pick(
            ["Throbbing", "Pulsating", "Pressing", "Stabbing", "Dull"],
            [0.45, 0.30, 0.10, 0.10, 0.05], n,
        ),
        "duration_hours": _clamp(np.random.normal(18, 8, n).round(1), 4, 72),
        "nausea": _pick([0, 1], [0.20, 0.80], n),
        "vomiting": _pick([0, 1], [0.55, 0.45], n),
        "photophobia": _pick([0, 1], [0.15, 0.85], n),
        "phonophobia": _pick([0, 1], [0.20, 0.80], n),
        "aura_present": np.zeros(n, dtype=int),
        "aura_type": np.full(n, "None"),
        "visual_disturbance": _pick([0, 1], [0.80, 0.20], n),
        "stress_level": _clamp(np.random.normal(7, 1.5, n).round(1), 1, 10),
        "sleep_hours": _clamp(np.random.normal(5.5, 1.5, n).round(1), 2, 12),
        "physical_activity": _pick(
            ["Sedentary", "Light", "Moderate", "Intense"],
            [0.35, 0.30, 0.25, 0.10], n,
        ),
        "caffeine_intake": _clamp(np.random.normal(3, 1.5, n).round(1), 0, 10),
        "alcohol_intake": _clamp(np.random.normal(2, 1.5, n).round(1), 0, 10),
        "weather_sensitivity": _pick([0, 1], [0.35, 0.65], n),
        "hormonal_factor": _pick([0, 1], [0.40, 0.60], n),
        "screen_time": _clamp(np.random.normal(8, 2.5, n).round(1), 0, 16),
        "frequency_per_month": _clamp(np.random.normal(5, 2, n).astype(int), 1, 15),
        "onset_pattern": _pick(
            ["Gradual", "Sudden", "With warning signs"],
            [0.40, 0.35, 0.25], n,
        ),
        "family_history": _pick([0, 1], [0.30, 0.70], n),
        "medication_response": _pick(
            ["Good", "Moderate", "Poor"],
            [0.30, 0.45, 0.25], n,
        ),
        "headache_type": np.full(n, "Migraine without aura"),
    }
    return pd.DataFrame(data)


def _generate_migraine_with_aura(n: int) -> pd.DataFrame:
    """Generate rows for Migraine with aura."""
    data = {
        "age": _clamp(np.random.normal(32, 10, n).astype(int), 12, 75),
        "gender": _pick(["Male", "Female", "Other"], [0.28, 0.67, 0.05], n),
        "pain_intensity": _clamp(np.random.normal(7.8, 1.0, n).round(1), 1, 10),
        "pain_location": _pick(
            ["Unilateral", "Bilateral", "Frontal", "Temporal", "Occipital"],
            [0.60, 0.10, 0.10, 0.15, 0.05], n,
        ),
        "pain_quality": _pick(
            ["Throbbing", "Pulsating", "Pressing", "Stabbing", "Dull"],
            [0.40, 0.35, 0.10, 0.10, 0.05], n,
        ),
        "duration_hours": _clamp(np.random.normal(16, 7, n).round(1), 4, 72),
        "nausea": _pick([0, 1], [0.15, 0.85], n),
        "vomiting": _pick([0, 1], [0.45, 0.55], n),
        "photophobia": _pick([0, 1], [0.10, 0.90], n),
        "phonophobia": _pick([0, 1], [0.15, 0.85], n),
        "aura_present": np.ones(n, dtype=int),
        "aura_type": _pick(
            ["Visual", "Sensory", "Motor", "Speech"],
            [0.55, 0.25, 0.10, 0.10], n,
        ),
        "visual_disturbance": _pick([0, 1], [0.15, 0.85], n),
        "stress_level": _clamp(np.random.normal(7.2, 1.5, n).round(1), 1, 10),
        "sleep_hours": _clamp(np.random.normal(5.5, 1.5, n).round(1), 2, 12),
        "physical_activity": _pick(
            ["Sedentary", "Light", "Moderate", "Intense"],
            [0.30, 0.30, 0.30, 0.10], n,
        ),
        "caffeine_intake": _clamp(np.random.normal(3.2, 1.5, n).round(1), 0, 10),
        "alcohol_intake": _clamp(np.random.normal(1.8, 1.2, n).round(1), 0, 10),
        "weather_sensitivity": _pick([0, 1], [0.30, 0.70], n),
        "hormonal_factor": _pick([0, 1], [0.35, 0.65], n),
        "screen_time": _clamp(np.random.normal(7.5, 2.5, n).round(1), 0, 16),
        "frequency_per_month": _clamp(np.random.normal(4, 2, n).astype(int), 1, 15),
        "onset_pattern": _pick(
            ["Gradual", "Sudden", "With warning signs"],
            [0.20, 0.20, 0.60], n,
        ),
        "family_history": _pick([0, 1], [0.25, 0.75], n),
        "medication_response": _pick(
            ["Good", "Moderate", "Poor"],
            [0.35, 0.40, 0.25], n,
        ),
        "headache_type": np.full(n, "Migraine with aura"),
    }
    return pd.DataFrame(data)


def _generate_tension_type(n: int) -> pd.DataFrame:
    """Generate rows for Tension-type headache."""
    data = {
        "age": _clamp(np.random.normal(38, 14, n).astype(int), 12, 75),
        "gender": _pick(["Male", "Female", "Other"], [0.42, 0.53, 0.05], n),
        "pain_intensity": _clamp(np.random.normal(4.5, 1.5, n).round(1), 1, 10),
        "pain_location": _pick(
            ["Unilateral", "Bilateral", "Frontal", "Temporal", "Occipital"],
            [0.10, 0.50, 0.20, 0.10, 0.10], n,
        ),
        "pain_quality": _pick(
            ["Throbbing", "Pulsating", "Pressing", "Stabbing", "Dull"],
            [0.05, 0.05, 0.50, 0.05, 0.35], n,
        ),
        "duration_hours": _clamp(np.random.normal(6, 4, n).round(1), 0.5, 48),
        "nausea": _pick([0, 1], [0.80, 0.20], n),
        "vomiting": _pick([0, 1], [0.95, 0.05], n),
        "photophobia": _pick([0, 1], [0.70, 0.30], n),
        "phonophobia": _pick([0, 1], [0.65, 0.35], n),
        "aura_present": np.zeros(n, dtype=int),
        "aura_type": np.full(n, "None"),
        "visual_disturbance": _pick([0, 1], [0.90, 0.10], n),
        "stress_level": _clamp(np.random.normal(7.5, 1.5, n).round(1), 1, 10),
        "sleep_hours": _clamp(np.random.normal(6, 1.5, n).round(1), 2, 12),
        "physical_activity": _pick(
            ["Sedentary", "Light", "Moderate", "Intense"],
            [0.40, 0.30, 0.20, 0.10], n,
        ),
        "caffeine_intake": _clamp(np.random.normal(4, 2, n).round(1), 0, 10),
        "alcohol_intake": _clamp(np.random.normal(1.5, 1.2, n).round(1), 0, 10),
        "weather_sensitivity": _pick([0, 1], [0.55, 0.45], n),
        "hormonal_factor": _pick([0, 1], [0.60, 0.40], n),
        "screen_time": _clamp(np.random.normal(9, 2.5, n).round(1), 0, 16),
        "frequency_per_month": _clamp(np.random.normal(10, 4, n).astype(int), 1, 30),
        "onset_pattern": _pick(
            ["Gradual", "Sudden", "With warning signs"],
            [0.60, 0.30, 0.10], n,
        ),
        "family_history": _pick([0, 1], [0.60, 0.40], n),
        "medication_response": _pick(
            ["Good", "Moderate", "Poor"],
            [0.50, 0.35, 0.15], n,
        ),
        "headache_type": np.full(n, "Tension-type headache"),
    }
    return pd.DataFrame(data)


def _generate_cluster(n: int) -> pd.DataFrame:
    """Generate rows for Cluster headache."""
    data = {
        "age": _clamp(np.random.normal(33, 8, n).astype(int), 18, 65),
        "gender": _pick(["Male", "Female", "Other"], [0.75, 0.20, 0.05], n),
        "pain_intensity": _clamp(np.random.normal(9.2, 0.6, n).round(1), 7, 10),
        "pain_location": _pick(
            ["Unilateral", "Bilateral", "Frontal", "Temporal", "Occipital"],
            [0.75, 0.05, 0.05, 0.10, 0.05], n,
        ),
        "pain_quality": _pick(
            ["Throbbing", "Pulsating", "Pressing", "Stabbing", "Dull"],
            [0.10, 0.10, 0.05, 0.65, 0.10], n,
        ),
        "duration_hours": _clamp(np.random.normal(1.5, 0.8, n).round(1), 0.25, 3),
        "nausea": _pick([0, 1], [0.50, 0.50], n),
        "vomiting": _pick([0, 1], [0.75, 0.25], n),
        "photophobia": _pick([0, 1], [0.40, 0.60], n),
        "phonophobia": _pick([0, 1], [0.45, 0.55], n),
        "aura_present": _pick([0, 1], [0.85, 0.15], n),
        "aura_type": _pick(
            ["None", "Visual", "Sensory", "Motor", "Speech"],
            [0.85, 0.08, 0.04, 0.02, 0.01], n,
        ),
        "visual_disturbance": _pick([0, 1], [0.55, 0.45], n),
        "stress_level": _clamp(np.random.normal(6, 2, n).round(1), 1, 10),
        "sleep_hours": _clamp(np.random.normal(5, 1.5, n).round(1), 2, 12),
        "physical_activity": _pick(
            ["Sedentary", "Light", "Moderate", "Intense"],
            [0.25, 0.30, 0.30, 0.15], n,
        ),
        "caffeine_intake": _clamp(np.random.normal(2.5, 1.5, n).round(1), 0, 10),
        "alcohol_intake": _clamp(np.random.normal(3.5, 2, n).round(1), 0, 10),
        "weather_sensitivity": _pick([0, 1], [0.50, 0.50], n),
        "hormonal_factor": _pick([0, 1], [0.75, 0.25], n),
        "screen_time": _clamp(np.random.normal(6, 2, n).round(1), 0, 16),
        "frequency_per_month": _clamp(np.random.normal(15, 5, n).astype(int), 3, 30),
        "onset_pattern": _pick(
            ["Gradual", "Sudden", "With warning signs"],
            [0.10, 0.75, 0.15], n,
        ),
        "family_history": _pick([0, 1], [0.80, 0.20], n),
        "medication_response": _pick(
            ["Good", "Moderate", "Poor"],
            [0.20, 0.35, 0.45], n,
        ),
        "headache_type": np.full(n, "Cluster headache"),
    }
    return pd.DataFrame(data)


# ── Main ─────────────────────────────────────────────────────────────────────
def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generate complete synthetic headache dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.

    Returns
    -------
    pd.DataFrame
        Combined dataset with all headache types.
    """
    counts = np.random.multinomial(n_samples, CLASS_WEIGHTS)
    generators = [
        _generate_migraine_without_aura,
        _generate_migraine_with_aura,
        _generate_tension_type,
        _generate_cluster,
    ]

    frames = [gen(c) for gen, c in zip(generators, counts)]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Introduce ~2 % missing values randomly (realistic data quality issue)
    mask = np.random.random(df.shape) < 0.02
    # Don't null out the target column
    target_col_idx = df.columns.get_loc("headache_type")
    mask[:, target_col_idx] = False
    df = df.mask(mask)

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Dataset generated: {OUTPUT_FILE}")
    print(f"   Shape : {df.shape}")
    print(f"   Classes:\n{df['headache_type'].value_counts().to_string()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
