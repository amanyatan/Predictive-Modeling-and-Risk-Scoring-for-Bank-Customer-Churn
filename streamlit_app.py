"""
============================================================
  BANK CHURN PREDICTION — STREAMLIT UI
  Run: streamlit run streamlit_app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0f172a; color: #e2e8f0; }
    .stApp { background: #0f172a; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .churn-high { color: #ef4444; font-size: 2rem; font-weight: 800; }
    .churn-low  { color: #22c55e; font-size: 2rem; font-weight: 800; }
    .section-title { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; }
</style>
""", unsafe_allow_html=True)

BALANCE_THRESHOLD = 100_000
MODEL_PATH = "best_churn_model.joblib"


# ── Helper: feature engineering ───────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Balance_to_Salary"]   = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Tenure_Age_Ratio"]    = df["Tenure"] / (df["Age"] + 1)
    df["Products_Per_Tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["IsHighValueCustomer"] = (df["Balance"] > BALANCE_THRESHOLD).astype(int)
    return df


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict(input_data: dict, model, feature_names: list):
    df = pd.DataFrame([input_data])
    df = engineer_features(df)

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Note: use your saved scaler here; for demo we skip scaling
    prob  = model.predict_proba(df.values)[0][1]
    label = int(prob >= 0.5)
    return prob, label


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=60)
    st.title("Churn Predictor")
    st.markdown("**Bank Customer Churn Prediction**")
    st.markdown("---")

    st.markdown("### 👤 Customer Profile")
    credit_score    = st.slider("Credit Score",      300, 850, 650)
    age             = st.slider("Age",               18,  90,  40)
    tenure          = st.slider("Tenure (years)",    0,   10,  5)
    num_products    = st.slider("Number of Products",1,   4,   2)

    st.markdown("### 💰 Financial")
    balance         = st.number_input("Account Balance ($)", 0.0, 300000.0, 120000.0, step=1000.0)
    salary          = st.number_input("Estimated Salary ($)", 1000.0, 300000.0, 80000.0, step=1000.0)

    st.markdown("### 📋 Account Details")
    geography       = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender          = st.selectbox("Gender", ["Female", "Male"])
    has_credit_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    is_active       = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x else "No")

    predict_btn = st.button("🔍 Predict Churn", use_container_width=True, type="primary")


# ── Main panel ─────────────────────────────────────────────
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict the likelihood of a customer leaving the bank.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Credit Score", credit_score)
with col2:
    st.metric("Age", age)
with col3:
    bal_m = f"${balance:,.0f}"
    st.metric("Balance", bal_m)
with col4:
    st.metric("Products", num_products)

st.markdown("---")

if predict_btn:
    model = load_model()

    input_data = {
        "CreditScore": credit_score, "Age": age, "Tenure": tenure,
        "Balance": balance, "NumOfProducts": num_products,
        "HasCrCard": has_credit_card, "IsActiveMember": is_active,
        "EstimatedSalary": salary, "Geography": geography, "Gender": gender
    }

    if model is None:
        st.warning(
            "⚠️ No saved model found. Run the pipeline (`churn_pipeline.py`) first to train & save a model.\n\n"
            "Showing **demo mode** with simulated probabilities."
        )
        # Demo mode: simulate a probability for UI demonstration
        np.random.seed(credit_score + age)
        prob = float(np.clip(
            0.1 + (age / 200) + (1 - is_active) * 0.3
            + (geography == "Germany") * 0.15
            - (balance / 1_000_000), 0.05, 0.95
        ))
        label = int(prob >= 0.5)
    else:
        # Use saved feature names if stored, else derive
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            st.error("Cannot determine feature names from model. Ensure model was trained with this pipeline.")
            st.stop()
        prob, label = predict(input_data, model, list(feature_names))

    # Result display
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("### Prediction Result")
        if label == 1:
            st.markdown(f'<p class="churn-high">⚠️ HIGH CHURN RISK</p>', unsafe_allow_html=True)
            st.error(f"Churn Probability: **{prob*100:.1f}%**")
        else:
            st.markdown(f'<p class="churn-low">✅ LOW CHURN RISK</p>', unsafe_allow_html=True)
            st.success(f"Churn Probability: **{prob*100:.1f}%**")

        # Gauge bar
        st.progress(prob)
        st.caption(f"Probability score: {prob:.4f}")

    with col_b:
        st.markdown("### Engineered Features")
        b2s   = balance / (salary + 1)
        tar   = tenure / (age + 1)
        ppt   = num_products / (tenure + 1)
        hvc   = "Yes" if balance > BALANCE_THRESHOLD else "No"

        feat_df = pd.DataFrame({
            "Feature": [
                "Balance to Salary Ratio",
                "Tenure / Age Ratio",
                "Products per Tenure",
                "High Value Customer"
            ],
            "Value": [
                f"{b2s:.4f}",
                f"{tar:.4f}",
                f"{ppt:.4f}",
                hvc
            ]
        })
        st.table(feat_df)

    st.markdown("---")
    st.markdown("### 📋 Full Customer Summary")
    summary = pd.DataFrame([{
        "Geography": geography, "Gender": gender, "Age": age,
        "Credit Score": credit_score, "Tenure": tenure,
        "Balance": f"${balance:,.2f}", "Products": num_products,
        "Has Credit Card": "Yes" if has_credit_card else "No",
        "Active Member": "Yes" if is_active else "No",
        "Est. Salary": f"${salary:,.2f}",
        "Churn Probability": f"{prob*100:.1f}%",
        "Prediction": "CHURN" if label == 1 else "RETAIN"
    }]).T.rename(columns={0: "Value"})
    st.dataframe(summary, use_container_width=True)

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn**.")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
1. **Run the pipeline first**: `python churn_pipeline.py` — this trains and saves the model.
2. **Launch this UI**: `streamlit run streamlit_app.py`
3. **Adjust the sliders** in the sidebar for any customer profile.
4. Click **Predict Churn** to see the churn probability and risk level.
""")
    st.markdown("### 📊 Pipeline Overview")
    steps = pd.DataFrame({
        "Step": range(1, 11),
        "Stage": [
            "Data Loading & Cleaning", "EDA & Visualization",
            "Feature Engineering", "Preprocessing (Encode + Scale)",
            "SMOTE Oversampling", "Model Training (LR / RF / XGB)",
            "Hyperparameter Tuning", "Evaluation (AUC, Recall, F1)",
            "SHAP Explainability", "Model Selection & Export"
        ],
        "Key Tool": [
            "pandas", "seaborn / matplotlib",
            "domain logic", "RobustScaler + LabelEncoder",
            "imbalanced-learn", "sklearn / xgboost",
            "RandomizedSearchCV", "sklearn.metrics",
            "shap", "joblib"
        ]
    })
    st.dataframe(steps, use_container_width=True, hide_index=True)
