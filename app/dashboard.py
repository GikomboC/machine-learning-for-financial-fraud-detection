
---

# Streamlit Dashboard (streamlit_app.py)

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Save your trained model as model.pkl

model = load_model()

# -----------------------------
# App Title
# -----------------------------
st.title("Financial Fraud Detection Dashboard")

st.markdown("Upload transaction data to detect fraudulent transactions.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# -----------------------------
# Prediction Function
# -----------------------------
def preprocess_input(df):
    df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["orig_zero_balance"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["dest_zero_balance"] = (df["oldbalanceDest"] == 0).astype(int)

    df["error_orig"] = df["amount"] - df["orig_balance_diff"]
    df["error_dest"] = df["amount"] - df["dest_balance_diff"]

    df = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")

    return df

# -----------------------------
# Run Predictions
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    try:
        processed_df = preprocess_input(df)

        predictions = model.predict(processed_df)
        probabilities = model.predict_proba(processed_df)[:, 1]

        df["Fraud_Prediction"] = predictions
        df["Fraud_Probability"] = probabilities

        st.subheader("Predictions")
        st.write(df.head())

        fraud_count = df["Fraud_Prediction"].sum()
        total = len(df)

        st.metric("Total Transactions", total)
        st.metric("Fraudulent Transactions", fraud_count)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Manual Input Section
# -----------------------------
st.subheader("Manual Transaction Input")

with st.form("manual_input"):
    step = st.number_input("Step", value=1)
    amount = st.number_input("Amount", value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", value=10000.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", value=9000.0)
    oldbalanceDest = st.number_input("Old Balance (Receiver)", value=5000.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", value=6000.0)

    type_option = st.selectbox(
        "Transaction Type",
        ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "step": step,
        "type": type_option,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    processed_input = preprocess_input(input_df)

    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1]

    st.write("Prediction:", "Fraud" if pred == 1 else "Not Fraud")
    st.write("Probability:", prob)
