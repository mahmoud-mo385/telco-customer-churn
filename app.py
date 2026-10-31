import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="",
    layout="centered",
)

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title(" Telco Customer Churn Prediction System")
st.write(
    "Enter customer details below to predict whether the customer is at risk"
    " of leaving the company."
)
st.markdown("---")
st.subheader(" Customer Information")
col1, col2 = st.columns(2)
with col1:
  gender = st.selectbox("Gender", ["Male", "Female"])
  senior_citizen = st.selectbox("Senior Citizen", [0, 1])
  partner = st.selectbox("Partner", ["Yes", "No"])
  dependents = st.selectbox("Dependents", ["Yes", "No"])
  tenure = st.slider(
      "Tenure (Months)", min_value=0, max_value=72, value=12
  )

with col2:
  phone_service = st.selectbox("Phone Service", ["Yes", "No"])
  contract = st.selectbox(
      "Contract Type", ["Month-to-month", "One year", "Two year"]
  )
  paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
  payment_method = st.selectbox(
      "Payment Method", [
          "Electronic check",
          "Mailed check",
          "Bank transfer (automatic)",
          "Credit card (automatic)",
      ]
  )
  monthly_charges = st.number_input(
      "Monthly Charges ($)", min_value=0.0, value=50.0
  )

total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
if st.button(" Predict Churn", use_container_width=True):
    input_data = pd.DataFrame({
        'gender': [1 if gender == 'Male' else 0],
        'SeniorCitizen': [senior_citizen],
        'Partner': [1 if partner == 'Yes' else 0],
        'Dependents': [1 if dependents == 'Dependents' else 0],  # أو حسب اختيارك
        'tenure': [tenure],
        'PhoneService': [1 if phone_service == 'Yes' else 0],
        'MultipleLines': [0],
        'InternetService': [0],
        'OnlineSecurity': [0],
        'OnlineBackup': [0],
        'DeviceProtection': [0],
        'TechSupport': [0],
        'StreamingTV': [0],
        'StreamingMovies': [0],
        'Contract': [0],
        'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
        'PaymentMethod': [0],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
})

    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        st.markdown("---")
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
          st.error(
              f" Warning: This customer is **at risk of churning**! (Churn"
              f" Probability: {prediction_proba[0][1]*100:.2f}%)"
          )
        else:
          st.success(
              f" This customer is **likely to stay** with the company."
              f" (Retention Probability: {prediction_proba[0][0]*100:.2f}%)"
          )

    except Exception as e:
        st.warning(
            " Please make sure all input columns match the original dataset"
            f" features. Detailed error: {e}"
        )