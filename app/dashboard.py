import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #111827;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -----------------------------
# Feature Engineering
# -----------------------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
        df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    if "newbalanceDest" in df.columns and "oldbalanceDest" in df.columns:
        df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    if "oldbalanceOrg" in df.columns:
        df["orig_zero_balance"] = (df["oldbalanceOrg"] == 0).astype(int)

    if "oldbalanceDest" in df.columns:
        df["dest_zero_balance"] = (df["oldbalanceDest"] == 0).astype(int)

    if "amount" in df.columns and "orig_balance_diff" in df.columns:
        df["error_orig"] = df["amount"] - df["orig_balance_diff"]

    if "amount" in df.columns and "dest_balance_diff" in df.columns:
        df["error_dest"] = df["amount"] - df["dest_balance_diff"]

    df = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")

    return df

# -----------------------------
# Helper: Summary Charts
# -----------------------------
def plot_prediction_distribution(df: pd.DataFrame):
    counts = df["Fraud_Prediction"].value_counts().sort_index()

    labels = ["Not Fraud", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title("Prediction Distribution")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_transaction_type_distribution(df: pd.DataFrame):
    if "type" not in df.columns:
        return

    counts = df["type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Transaction Type Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Transaction Type")
    plt.xticks(rotation=30)
    st.pyplot(fig)

def plot_fraud_probability_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Fraud_Probability"], bins=30)
    ax.set_title("Fraud Probability Distribution")
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# -----------------------------
# Helper: SHAP
# -----------------------------
def get_feature_names_from_pipeline(_model, X_processed: pd.DataFrame):
    try:
        preprocessor = _model.named_steps["preprocessor"]
        classifier = _model.named_steps["classifier"]

        transformed = preprocessor.transform(X_processed)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

        return classifier, transformed, feature_names
    except Exception:
        return None, None, None

def show_shap_explainability(_model, X_input: pd.DataFrame):
    classifier, transformed, feature_names = get_feature_names_from_pipeline(_model, X_input)

    if classifier is None:
        st.warning("SHAP could not be generated for this model structure.")
        return

    try:
        sample_size = min(200, len(X_input))
        transformed_sample = transformed[:sample_size]

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed_sample)

        st.subheader("SHAP Explainability")

        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_array = shap_values[1]
        else:
            shap_array = shap_values

        fig_summary = plt.figure()
        shap.summary_plot(
            shap_array,
            transformed_sample,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig_summary, clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP generation failed: {e}")

# -----------------------------
# Header
# -----------------------------
st.title("Financial Fraud Detection Dashboard")
st.markdown("Upload transaction data or enter a transaction manually to predict fraud risk.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.50, 0.01)
show_shap = st.sidebar.checkbox("Enable SHAP Explainability", value=False)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Batch Prediction", "Manual Prediction"])

# -----------------------------
# Batch Prediction Tab
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        try:
            processed_df = preprocess_input(df)

            probabilities = model.predict_proba(processed_df)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            results_df = df.copy()
            results_df["Fraud_Probability"] = probabilities
            results_df["Fraud_Prediction"] = predictions

            total_transactions = len(results_df)
            fraud_count = int(results_df["Fraud_Prediction"].sum())
            non_fraud_count = total_transactions - fraud_count
            fraud_rate = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{total_transactions:,}")
            col2.metric("Fraud Predictions", f"{fraud_count:,}")
            col3.metric("Non-Fraud Predictions", f"{non_fraud_count:,}")
            col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            st.subheader("Prediction Results")
            st.dataframe(results_df.head(20), use_container_width=True)

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                plot_prediction_distribution(results_df)

            with chart_col2:
                plot_fraud_probability_histogram(results_df)

            st.subheader("Transaction Insights")
            plot_transaction_type_distribution(results_df)

            suspicious_df = results_df[results_df["Fraud_Prediction"] == 1].sort_values(
                by="Fraud_Probability", ascending=False
            )

            st.subheader("Most Suspicious Transactions")
            st.dataframe(suspicious_df.head(20), use_container_width=True)

            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

            if show_shap:
                show_shap_explainability(model, processed_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------------
# Manual Prediction Tab
# -----------------------------
with tab2:
    st.subheader("Enter Transaction Details")

    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            step = st.number_input("Step", min_value=1, value=1)
            type_option = st.selectbox(
                "Transaction Type",
                ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"]
            )
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            oldbalanceOrg = st.number_input("Old Balance - Sender", min_value=0.0, value=10000.0)

        with col2:
            newbalanceOrig = st.number_input("New Balance - Sender", min_value=0.0, value=9000.0)
            oldbalanceDest = st.number_input("Old Balance - Receiver", min_value=0.0, value=5000.0)
            newbalanceDest = st.number_input("New Balance - Receiver", min_value=0.0, value=6000.0)
            isFlaggedFraud = st.selectbox("Flagged Fraud", [0, 1], index=0)

        submitted = st.form_submit_button("Predict Fraud Risk")

    if submitted:
        input_df = pd.DataFrame([{
            "step": step,
            "type": type_option,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": isFlaggedFraud
        }])

        try:
            processed_input = preprocess_input(input_df)
            prob = model.predict_proba(processed_input)[0][1]
            pred = int(prob >= threshold)

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Prediction", "Fraud" if pred == 1 else "Not Fraud")
            result_col2.metric("Fraud Probability", f"{prob:.4f}")

            st.dataframe(input_df, use_container_width=True)

            if show_shap:
                show_shap_explainability(model, processed_input)

        except Exception as e:
            st.error(f"Manual prediction failed: {e}")
