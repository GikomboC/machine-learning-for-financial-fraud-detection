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

### Challenges
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
| isFraud | Target variable (0 = Legit, 1 = Fraud) |
| isFlaggedFraud | Flag for large illegal transactions |

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
- `orig_balance_diff`
- `dest_balance_diff`
- `orig_zero_balance`
- `dest_zero_balance`
- `error_orig`
- `error_dest`

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

> Replace with actual results after running the notebook.

---

## Key Insights

- Fraud is concentrated in specific transaction types (TRANSFER, CASH-OUT)
- Balance inconsistencies are strong fraud indicators
- Engineered features significantly improve model performance
- Random Forest outperforms Logistic Regression

---

## Project Structure


```text
fraud-detection/
├── data/
│   └── financial_fraud_detection.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── app/
│   └── streamlit_app.py
├── outputs/
│   └── fraud_predictions.csv
├── README.md
└── requirements.txt
```

## Streamlit Dashboard Features
- Upload transaction data
- Real-time fraud prediction
- Probability scoring
- Interactive results display

## Future Improvements
- XGBoost / LightGBM models
- SMOTE for imbalance handling
- Hyperparameter tuning
- SHAP explainability
- Real-time API deployment
## Technologies Used
- python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Streamlit


## Author
# Caleb Gikombo
# Mechatronics Engineer | Data Scientist

# License
This project is open-source and available under the MIT License.
