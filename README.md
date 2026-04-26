# Bank Customer Churn Prediction Pipeline

This repository contains an end-to-end machine learning pipeline to predict bank customer churn.

## Features implemented:
- **Data Preprocessing:** Handles missing values, performs One-Hot Encoding and Label Encoding, and drops irrelevant columns.
- **Feature Engineering:** Creates powerful predictive features like `Balance_to_Salary`, `Tenure_Age_Ratio`, `Products_Per_Tenure`, and `IsHighValueCustomer`.
- **Class Imbalance Handling:** Uses SMOTE to balance the training data.
- **Model Tuning & Selection:** Trains Logistic Regression, Random Forest, and XGBoost with RandomizedSearchCV optimized for ROC-AUC.
- **Explainability:** Calculates Feature Importances and SHAP values.
- **Streamlit Web UI:** A beautiful and simple interface to test predictions in real-time.

## Instructions:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Your Dataset:**
   Place your dataset (`Churn_Modelling.csv`) in the same directory as the scripts.

3. **Train the Model:**
   ```bash
   python churn_pipeline.py
   ```
   *This will output EDA visualizations, model evaluation metrics, SHAP plots, and save the trained pipeline to `churn_pipeline.pkl`.*

4. **Run the Streamlit Interface:**
   ```bash
   streamlit run app.py
   ```
