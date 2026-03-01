# Headache Pattern Analysis & Migraine Type Prediction
## Project Report

---

## 1. Problem Statement

Headaches and migraines are among the most prevalent neurological conditions worldwide, affecting approximately 50% of the global population. Distinguishing between different headache types—such as migraine without aura, migraine with aura, tension-type headache, and cluster headache—is crucial for appropriate management and treatment planning. However, accurate classification often requires specialized clinical expertise that may not be readily available, especially in primary care settings.

This project addresses the need for an accessible, AI-based decision-support system that can analyze headache symptoms, behavioral patterns, environmental triggers, and clinical features to predict the most likely headache/migraine type. The system is intended for **early awareness and educational purposes only** and does not serve as a medical diagnostic tool.

**Target Classes:**
1. Migraine without aura
2. Migraine with aura
3. Tension-type headache
4. Cluster headache

---

## 2. Literature Survey

### 2.1 Headache Classification Standards
The International Classification of Headache Disorders (ICHD-3), published by the International Headache Society, provides the gold standard for headache classification. It categorizes headaches into primary headaches (migraine, tension-type, cluster, and other trigeminal autonomic cephalalgias) and secondary headaches caused by underlying conditions (Headache Classification Committee of the International Headache Society, 2018).

### 2.2 Machine Learning in Headache Diagnosis
Several studies have explored machine learning approaches for headache classification:

- **Katsuki et al. (2020)** demonstrated that Random Forest classifiers could differentiate between migraine and tension-type headache with accuracy exceeding 85% using clinical questionnaire data.
- **Vandenbussche et al. (2021)** applied Support Vector Machines to classify headache types using electronic health record features, achieving F1-scores above 0.80 for migraine detection.
- **Garcia-Chimeno et al. (2017)** used decision tree and ensemble methods on patient-reported symptom data, showing that automated classification can approach the accuracy of clinical experts.
- **Chiang et al. (2019)** explored deep learning approaches for headache classification using patient diary data, demonstrating the potential of neural networks for temporal pattern recognition.

### 2.3 Feature Importance in Headache Classification
Key discriminating features identified in the literature include:
- Pain location (unilateral vs. bilateral)
- Pain quality (throbbing vs. pressing)
- Duration and frequency of episodes
- Associated symptoms (nausea, photophobia, phonophobia)
- Presence and type of aura
- Lifestyle and environmental triggers

---

## 3. Methodology

### 3.1 Problem Formulation
This is a **multi-class classification problem** where the input consists of 25+ features describing a patient's headache characteristics, associated symptoms, triggers, and clinical history. The output is one of four headache types.

### 3.2 Dataset Generation
Since publicly available, well-labeled headache datasets are scarce, we generated a **synthetic dataset of 5,000 records** with clinically-informed probability distributions for each headache type. The distributions were designed to reflect real-world epidemiological patterns:

- **Class distribution**: Migraine without aura (35%), Tension-type (35%), Migraine with aura (20%), Cluster headache (10%)
- **Gender bias**: Cluster headaches show male predominance (75%); migraines show female predominance (65%)
- **Pain characteristics**: Cluster headaches have highest intensity (mean 9.2/10); tension-type has lowest (mean 4.5/10)
- **Duration patterns**: Cluster headaches are short (0.25–3 hours); migraines are long (4–72 hours)

A 2% random missing value rate was introduced to simulate real-world data quality issues.

### 3.3 Data Preprocessing Pipeline

1. **Missing Value Handling**: Median imputation for numeric features; mode imputation for categorical and binary features.
2. **Feature Engineering**: Four derived features were created:
   - **Severity Score**: Weighted combination of pain intensity and associated symptoms
   - **Frequency Index**: Normalized headache frequency × duration
   - **Trigger Count**: Number of elevated lifestyle triggers
   - **Symptom Count**: Total associated symptoms present
3. **Encoding**: Label encoding for ordinal/categorical features
4. **Scaling**: StandardScaler normalization (fit on training set only)
5. **Train/Test Split**: 80/20 stratified split

### 3.4 Model Selection
Three classification algorithms were selected:

1. **Random Forest**: Ensemble of decision trees with bagging; robust to overfitting and handles non-linear relationships well.
2. **Support Vector Machine (SVM)**: Finds optimal hyperplane for class separation; effective in high-dimensional spaces.
3. **Gradient Boosting**: Sequential ensemble method that builds trees to correct residual errors; typically achieves high accuracy.

### 3.5 Hyperparameter Tuning
5-fold cross-validated Grid Search was used to optimize each model:

| Model | Parameters Tuned |
|-------|-----------------|
| Random Forest | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| SVM | C, kernel, gamma |
| Gradient Boosting | n_estimators, max_depth, learning_rate, subsample |

---

## 4. System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│              HTML + CSS (Medical-themed UI)                  │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│   │ Symptom Form│  │ Predict Btn  │  │  Results Panel  │   │
│   └─────────────┘  └──────────────┘  └─────────────────┘   │
└────────────────────────┬───────────────────────────────────┘
                         │ POST /predict
                         ▼
┌────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND                            │
│                      (app.py)                               │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│   │ Input Parser │→ │ Preprocessor │→ │ Model Predictor│   │
│   └──────────────┘  └──────────────┘  └────────────────┘   │
│                                              │              │
│                                              ▼              │
│                                     ┌────────────────┐      │
│                                     │ Risk Assessor  │      │
│                                     └────────────────┘      │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                  ML PIPELINE                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Dataset Gen │→ │ Preprocessing│→ │  Model Training    │ │
│  │ (Synthetic) │  │ & Feature Eng│  │  (RF, SVM, GBM)    │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│                                              │              │
│                                              ▼              │
│                                     ┌────────────────────┐  │
│                                     │ Evaluation & Plots │  │
│                                     └────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Algorithm Explanation

### 5.1 Random Forest
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the individual trees' predictions. Key advantages:
- Handles non-linear relationships
- Resistant to overfitting
- Provides feature importance scores
- Works well with both numerical and categorical features

### 5.2 Support Vector Machine (SVM)
SVM finds the optimal hyperplane that maximizes the margin between classes. For multi-class problems, it uses the one-vs-one strategy. With the RBF kernel, it can model non-linear decision boundaries by mapping features into a higher-dimensional space.

### 5.3 Gradient Boosting
Gradient Boosting builds an ensemble of weak learners (decision trees) sequentially, where each new tree corrects the errors of the previous ones. It minimizes a loss function using gradient descent in function space, achieving high accuracy through iterative refinement.

---

## 6. Results Analysis

### 6.1 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Random Forest | 98.50% | 98.50% | 98.50% | 98.50% | 92.6s |
| **SVM** | **98.90%** | **98.91%** | **98.90%** | **98.90%** | **32.4s** |
| Gradient Boosting | 98.30% | 98.31% | 98.30% | 98.30% | 501.1s |

### 6.2 Best Model: SVM (RBF Kernel)
The SVM classifier with RBF kernel (C=1, gamma=scale) achieved the highest F1-score of 98.90%.

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Migraine without aura | 0.98 | 0.99 | 0.99 | 343 |
| Migraine with aura | 1.00 | 0.99 | 0.99 | 203 |
| Tension-type headache | 0.99 | 0.99 | 0.99 | 359 |
| Cluster headache | 1.00 | 0.97 | 0.98 | 95 |

### 6.3 Key Observations
1. All three models achieved accuracy above 98%, indicating the engineered features effectively capture the distinguishing characteristics of each headache type.
2. SVM achieved the best overall balance of accuracy and training time.
3. Cluster headache had slightly lower recall (0.97), likely due to its smaller representation in the dataset (10%).
4. Migraine with aura achieved perfect precision (1.00), suggesting the aura-related features are highly discriminative.

---

## 7. Future Scope

1. **Real-World Dataset**: Integrate real clinical data from hospitals or headache clinics to validate and improve model generalization.
2. **Deep Learning**: Explore neural network architectures (e.g., TabNet, Transformers) for potentially higher accuracy on complex feature interactions.
3. **Temporal Analysis**: Implement time-series analysis of headache diaries to identify patterns and predict upcoming episodes.
4. **Mobile Application**: Develop a mobile app for daily headache logging and real-time prediction.
5. **Explainable AI**: Integrate SHAP (SHapley Additive exPlanations) values for better model interpretability and patient-friendly explanations.
6. **Multi-language Support**: Add multilingual support for broader accessibility.
7. **Integration with EHR**: Connect with Electronic Health Record systems for seamless data flow.
8. **FDA/CDSCO Compliance**: Pursue regulatory compliance if advancing toward clinical use.

---

## 8. Conclusion

This project successfully demonstrates the application of machine learning in headache pattern analysis and migraine type prediction. The AI-based decision-support system:

- Achieves **98.9% accuracy** in classifying four types of headaches using an SVM classifier
- Provides an intuitive web interface for symptom input and real-time prediction
- Includes probability-based risk scoring and confidence assessment
- Maintains clear medical disclaimers to ensure responsible use

The system serves as a valuable educational and awareness tool that can help individuals understand their headache patterns and make informed decisions about seeking professional medical advice. While not a replacement for clinical diagnosis, it demonstrates the potential of AI in supporting early identification and management of headache disorders.

---

## Deployment Guide

### Option 1: Local Deployment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python -m ml.generate_dataset

# 3. Train models
python -m ml.model_training

# 4. Run the web app
python app.py
# Open http://127.0.0.1:5000
```

### Option 2: Deploy on Render
1. Push project to a GitHub repository
2. Create a `render.yaml` or connect the repo on [Render](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add `gunicorn` to `requirements.txt`

### Option 3: Deploy on Railway
1. Push project to GitHub
2. Connect the repo on [Railway](https://railway.app)
3. Railway auto-detects the Flask app and deploys
4. Set `PORT` environment variable if needed

---

**Disclaimer**: This project is developed for educational and research purposes as part of a B.Tech final-year project. The system is NOT intended for medical diagnosis. Always consult a qualified healthcare professional for headache evaluation and treatment.
