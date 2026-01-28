import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="centered"
)

st.title("Loan Default Prediction System")
st.write(
    "Upload a test dataset and the trained model will predict "
    "loan default risk for each applicant."
)

# LOAD MODEL
model = joblib.load("model/loan_default_model.pkl")
features = list(joblib.load("model/model_features.pkl"))

st.success("Model loaded successfully")

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV file)",
    type=["csv"]
)

if uploaded_file is not None:

    # Read uploaded dataset
    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(test_df.head())

    # ALIGN FEATURES
    test_df_aligned = test_df.reindex(columns=features, fill_value=0)

    # PREDICT PROBABILITIES
    probabilities = model.predict_proba(test_df_aligned)[:, 1]

    # APPLY BUSINESS THRESHOLD
    threshold = 0.30  # Custom threshold for imbalanced data
    predictions = (probabilities >= threshold).astype(int)

    #CREATE RESULT DATAFRAME
    result_df = test_df.copy()
    result_df["Prediction"] = predictions
    result_df["Default_Probability"] = probabilities

    st.subheader("Prediction Results")
    st.dataframe(result_df.head())