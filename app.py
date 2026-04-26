import streamlit as st
import pandas as pd
from churn_pipeline import predict_customer_churn

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")
st.markdown("""
Predict whether a customer is likely to churn based on their demographic and account information. 
Adjust the inputs below to see the real-time churn probability.
""")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    has_cr_card = st.selectbox("Has Credit Card?", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    is_active_member = st.selectbox("Is Active Member?", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0)

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    try:
        churn_prob = predict_customer_churn(input_data)
        
        st.divider()
        st.subheader("Prediction Results")
        
        # Display Progress Bar for Probability
        st.progress(float(churn_prob))
        
        if churn_prob > 0.5:
            st.error(f"⚠️ High Risk of Churn! (Probability: {churn_prob:.1%})")
            st.markdown("**Recommendation:** Proactively reach out to the customer with retention offers.")
        else:
            st.success(f"✅ Low Risk of Churn. (Probability: {churn_prob:.1%})")
            st.markdown("**Recommendation:** Continue maintaining a good relationship with the customer.")
            
    except FileNotFoundError:
        st.error("Model pipeline not found! Please run `python churn_pipeline.py` first to train and save the model.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
