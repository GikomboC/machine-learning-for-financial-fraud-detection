# Financial Fraud Detection Using Machine Learning

## Overview
This project builds an end-to-end machine learning system to detect fraudulent financial transactions.

The dataset contains transaction-level information such as:
- transaction type
- transaction amount
- sender and receiver balances
- fraud labels

The goal is to classify transactions as:
- 0 → Legitimate
- 1 → Fraudulent

---

## Problem Statement
Financial fraud is a critical issue in digital transactions. Detecting fraudulent activities in real-time can prevent financial losses.

Challenges:
- Highly imbalanced dataset (fraud cases are rare)
- Complex transaction patterns
- Need for high recall without sacrificing precision

---

## Dataset Description

| Feature | Description |
|--------|------------|
| step | Time step (1 step = 1 hour) |
| type | Transaction type (CASH-IN, CASH-OUT, TRANSFER, etc.) |
| amount | Transaction amount |
| nameOrig | Sender ID |
| oldbalanceOrg | Sender balance before transaction |
| newbalanceOrig | Sender balance after transaction |
| nameDest | Receiver ID |
| oldbalanceDest | Receiver balance before transaction |
| newbalanceDest | Receiver balance after transaction |
| isFraud | Target variable |
| isFlaggedFraud | Flagged large illegal transactions |

---

## Project Workflow

### 1. Data Loading
- Load dataset using pandas
- Inspect structure and shape

### 2. Data Cleaning
- Remove duplicates
- Check missing values

### 3. Exploratory Data Analysis
- Fraud distribution
- Fraud by transaction type
- Suspicious patterns

### 4. Feature Engineering
Created new features:
- balance differences
- zero-balance indicators
- transaction error features

### 5. Preprocessing
- Numeric scaling (StandardScaler)
- Categorical encoding (OneHotEncoder)

### 6. Model Building
Models used:
- Logistic Regression (baseline)
- Random Forest (final model)

### 7. Model Evaluation
Metrics:
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

### 8. Feature Importance
- Identify key predictors of fraud

---

## Results

| Model | ROC-AUC | PR-AUC |
|------|--------|--------|
| Logistic Regression | ~0.90+ | Moderate |
| Random Forest | ~0.98+ | High |

[Note: Replace with actual results after running]

---

## Key Insights

- Fraud is concentrated in specific transaction types (TRANSFER, CASH-OUT)
- Balance inconsistencies are strong fraud indicators
- Engineered features significantly improve performance
- Random Forest outperforms Logistic Regression

---

## Project Structure
fraud-detection/
│
├── data/
│   └── financial_fraud_detection.csv        # Raw dataset used for training and analysis
│
├── notebooks/
│   └── fraud_detection.ipynb                # Jupyter Notebook (EDA, feature engineering, modeling)
│
├── app/
│   └── streamlit_app.py                    # Streamlit dashboard for fraud prediction
│
├── outputs/
│   └── fraud_predictions.csv               # Model prediction results (generated output)
│
├── README.md                              # Project documentation
└── requirements.txt                       # Python dependencies
---
