# 🏦 Bank Customer Churn AI Pipeline
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/ML-XGBoost%20%7C%20Sklearn-green)](https://xgboost.readthedocs.io/)
[![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io/)

Predictive analytics system designed to identify high-risk customers likely to churn, leveraging advanced machine learning, automated feature engineering, and a premium interactive dashboard.

---

## 🚀 The System Architecture

The project is structured as an end-to-end ML pipeline that transforms raw banking data into actionable insights:

1.  **Ingestion & Cleaning**: Automated handling of missing values and removal of non-predictive features (IDs, Row Numbers).
2.  **Preprocessing Engine**: 
    *   **Label Mapping**: Binary encoding for categorical gender data.
    *   **One-Hot Encoding**: Geometric mapping for geographical regions.
    *   **Robust Scaling**: Uses `RobustScaler` to normalize financial distributions while remaining resistant to extreme outliers.
3.  **Feature Engineering**: Creates derived metrics like *Balance-to-Salary Ratio* and *Tenure-to-Age Ratio* to capture complex behavioral patterns.
4.  **Imbalance Management**: Applies **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure the model learns effectively from churn events.
5.  **Ensemble Modeling**: Orchestrates a tournament between Logistic Regression, Random Forest, and XGBoost to find the most accurate predictor.

---

## 📊 Model Performance

After rigorous training and hyperparameter tuning using `RandomizedSearchCV`, the **XGBoost** model emerged as the champion.

### Champion Model: **XGBoost Classifier**
| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **85.75%** | High overall classification precision |
| **ROC-AUC** | **0.8668** | Excellent ability to distinguish between churners and retainers |
| **Recall** | **64.13%** | Effectively captures over 64% of potential churners |
| **F1-Score** | **0.6468** | Balanced harmonic mean of Precision and Recall |

---

## 🛠️ Installation & Setup

1.  **Clone the Environment**:
    Ensure you have Python 3.8+ installed, then install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare the Data**:
    Ensure your dataset (`European_Bank.csv`) is placed in the root directory.

---

## 🕹️ How to Run

### 1. Execute the Pipeline
Train the models, generate explainability plots (SHAP), and export the champion pipeline:
```bash
python churn_pipeline.py
```
*Outputs: `churn_pipeline.pkl`, EDA visualizations in `eda_outputs/`, and metrics in `model_outputs/`.*

### 2. Launch the Premium Dashboard
Run the high-end interactive UI for real-time customer analysis:
```bash
streamlit run streamlit_app.py
```

---

## 🎨 Interactive Dashboard Features
*   **Real-time Risk Scoring**: Instant churn probability for any customer profile.
*   **Interactive Gauges**: Visual risk metering using Plotly.
*   **Pattern Insights**: Automated loyalty and financial exposure indexing.
*   **AI Recommendations**: Strategic advice tailored to the customer's risk level.

---

## 📂 Project Structure
*   `churn_pipeline.py`: The core ML engine (Training, Scaling, Feature Engineering).
*   `streamlit_app.py`: The premium Glassmorphism-style dashboard.
*   `requirements.txt`: Project dependencies.
*   `model_outputs/`: ROC curves, Feature Importance, and SHAP summaries.
*   `eda_outputs/`: Distribution plots and correlation heatmaps.

---