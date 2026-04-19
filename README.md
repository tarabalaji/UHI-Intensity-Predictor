# UHIQ – Urban Heat Island Prediction System

UHIQ is a Streamlit-based modeling tool that predicts Urban Heat Island (UHI) intensity using two parallel approaches:

1. A **physics-inspired linear model derived from environmental relationships**
2. A **machine learning model (XGBoost regression)** trained on urban climate data

The system allows users to modify urban environmental variables and immediately observe their effect on predicted UHI intensity.

---

## Core Idea

Urban Heat Island intensity is strongly influenced by land surface characteristics such as vegetation density, elevation variation, and land cover change.

UHIQ models this relationship using:

- A simplified interpretable equation derived from linear regression coefficients
- A nonlinear ML model trained on the same feature space for comparison

This allows direct evaluation of:
- interpretability vs predictive power
- physical assumptions vs learned patterns

---

## Inputs

The model uses the following features:

- `NDVI_urb_CT_act` → urban vegetation index
- `DelNDVI_annual` → vegetation difference between urban and rural areas
- `DelDEM` → elevation difference (scaled by 1/100 internally)
- `UHI_annual_day` → observed target variable

Optional simulation inputs:
- NDVI (vegetation level)
- ΔNDVI (vegetation contrast)
- ΔDEM (elevation change)
- Albedo
- Urban fraction

---

## Models

### 1. Formula-Based Model (Linear Regression)

The model learns coefficients from data and is expressed as:

UHI = intercept + α(1 − NDVI) + β(ΔNDVI) + γ(ΔDEM)

This version is used for:
- interpretability
- sensitivity analysis
- comparison against ML model

---

### 2. Machine Learning Model (XGBoost)

An XGBoost regressor is trained on the same input features:

- NDVI
- ΔNDVI
- ΔDEM

Key parameters:
- 1000 estimators
- learning_rate = 0.03
- max_depth = 4
- subsample = 0.85
- regularization (L1 + L2 enabled)

This model captures nonlinear interactions between variables.

---

## Features of the App

### City-Level Data Exploration
- Select any urban area from dataset
- View corresponding environmental and UHI values

### Real-Time Simulation
Users can modify environmental variables and immediately see:
- Formula model prediction
- ML model prediction
- Actual observed UHI

### Sensitivity Analysis
- NDVI is varied across range [0,1]
- Effect on UHI is computed for both models
- Results visualized as line charts

### Model Comparison
- Side-by-side comparison of:
  - Actual UHI
  - Formula prediction
  - ML prediction

### Model Evaluation
- Mean Absolute Error (MAE) is computed for both models
- Lower MAE indicates better predictive accuracy

### Feature Importance
- Permutation importance is used to evaluate feature influence in ML model

### Performance Tracking
- Tracks interaction runtime per session
- Computes average response time across interactions

---

## Output Visualizations

- UHI vs NDVI sensitivity curves (Formula + ML)
- Actual vs predicted bar comparisons
- Feature importance ranking
- Error comparison (MAE chart)

---

## How to Run

### Install dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn xgboost openpyxl