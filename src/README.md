# Telecom Churn Classification - End To End Pipeline with Deployment

## Project Overview

This project implements a pipeline for predicting telecom customer churn

Below is the structured content:
- Formal hypothesis testing (Accept/Reject H₀)
- Maximum Likelihood Estimation (MLE)
- Pseudo R² statistical validation
- Proper class imbalance handling (SMOTE inside CV)
- Bayesian hyperparameter tuning
- Probability calibration (Platt Scaling)
- Clean modular architecture
- Production-grade FastAPI deployment
- Dockerized serving

This repository fulfills statistical modeling and ML system design.

---

# Business Problem

Telecom companies lose significant revenue due to customer churn.

Objective:
Predict whether a customer will churn based on demographic, service, and billing attributes.

Target variable:
```
Churn (Yes / No)
```

---

# Statistical Methodology

This pipeline does not rely purely on feature importance from tree models but uses clean layered statistical validation:

---

## Hypothesis-Driven Feature Selection

### Numerical Features → Welch’s T-Test

Null Hypothesis:

H₀: Mean(feature | churn=Yes) = Mean(feature | churn=No)

If p-value < 0.05 → Reject H₀ → Feature retained  
Else → Fail to reject H₀ → Feature removed

---

### Categorical Features → Chi-Square Test

Null Hypothesis:

H₀: Feature ⟂ Churn (independent)

If p-value < 0.05 → Reject H₀ → Feature retained  
Else → Fail to reject H₀ → Feature removed

---

## MLE-Based Logistic Regression

Logistic regression is estimated using Maximum Likelihood Estimation:

L(β) = ∏ p_i^y_i (1 − p_i)^(1 − y_i)

Each feature coefficient is tested:

H₀: βᵢ = 0  
H₁: βᵢ ≠ 0  

Features with p-value < 0.05 are retained.

---

## McFadden’s Pseudo R²

Since churn is binary, traditional R² is invalid.

We use:

R² = 1 − (LL_model / LL_null)

Result obtained:

Pseudo R² = 0.2845

For behavioral churn prediction, this represents strong explanatory power.

---

# Model Pipeline

Architecture:

Train/Test Split (80/20, Stratified)
        ↓
ColumnTransformer
    ├── OneHotEncoder (drop_first=True in order to avoid dummy variable trap)
    └── Numeric passthrough
        ↓
SMOTE (inside cross-validation folds)
        ↓
RandomForestClassifier
        ↓
RandomizedSearchCV (Bayesian tuning)
        ↓
CalibratedClassifierCV (Platt scaling)
        ↓
Evaluation (ROC-AUC)

---

# Final Model Performance

ROC-AUC: 0.8331

For telecom churn datasets, ROC-AUC > 0.80 is considered strong.

---

# Clean Architecture

```
pfizer_telecom_cls/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   ├── config.py                # Global configurations
│   ├── data_loader.py           # Data ingestion & cleaning
│   ├── statistical_tests.py     # T-tests & Chi-square tests
│   ├── mle_feature_selection.py # Logistic MLE + Pseudo R²
│   ├── model_builder.py         # ML pipeline with SMOTE & tuning
│   └── save_model.py            # Model persistence
│
├── train_pipeline.py            # End-to-end training entrypoint
│
├── api/
│   ├── main.py                  # FastAPI inference app
│   ├── schemas.py               # Request validation schema
│   └── load_model.py            # Model loading
│
├── requirements.txt
└── Dockerfile
```

Design principles:
- Clear separation of concerns
- Statistical modules separated from modeling modules
- Training and serving decoupled
- Production inference isolated in API layer

---

# Running the Project

## Create Virtual Environment

Windows:
```
python -m venv py312_pfizer_tel_cls
py312_pfizer_tel_cls\Scripts\activate
```

Mac/Linux:
```
python3 -m venv py312_pfizer_tel_cls
source py312_pfizer_tel_cls/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

---

## Train Model

```
python train_pipeline.py
```

Outputs:
```
churn_model.pkl
```

---

## Run API

```
uvicorn api.main:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

---

# Docker Deployment

Build image:
```
docker build -t telecom-churn .
```

Run container:
```
docker run -p 8000:8000 telecom-churn
```

---

This implementation:
- Uses formal null hypothesis testing
- Applies MLE coefficient validation
- Quantifies explanatory power using Pseudo R²
- Places SMOTE correctly inside CV folds
- Calibrates probabilities for reliable decision thresholds
- Maintains clean production architecture

---

# Key Technical Highlights

- Maximum Likelihood Estimation (statsmodels)
- Likelihood-based feature significance
- McFadden’s Pseudo R²
- SMOTE integrated correctly in pipeline
- Cross-validation-aware oversampling
- Bayesian hyperparameter tuning
- Probability calibration (Platt scaling)
- Modular ML engineering design
- Production FastAPI deployment

---