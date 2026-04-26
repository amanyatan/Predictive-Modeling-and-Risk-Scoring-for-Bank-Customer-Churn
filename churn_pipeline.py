import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             RocCurveDisplay, classification_report)
from imblearn.over_sampling import SMOTE

class BankChurnPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.preprocessor = None
        self.best_model_name = None
        self.label_encoders = {}
        
    def load_data(self):
        """1. Load dataset from CSV and perform basic cleanup."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Drop irrelevant columns
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Year']
        self.df = self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], errors='ignore')
        
        # Handle missing values properly
        if self.df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            for col in self.df.select_dtypes(include=np.number).columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            for col in self.df.select_dtypes(include=['object']).columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                
        return self.df

    def perform_eda(self):
        """Perform Exploratory Data Analysis."""
        print("\n--- EDA Summary ---")
        print(self.df.describe())
        
        # Distributions and Correlation (Saving to disk)
        os.makedirs("eda_outputs", exist_ok=True)
        
        # Target Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x='Exited', palette='viridis')
        plt.title("Churn Class Distribution")
        plt.savefig("eda_outputs/target_distribution.png")
        plt.close()

        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        num_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.savefig("eda_outputs/correlation_heatmap.png")
        plt.close()
        print("EDA visualizations saved to 'eda_outputs/' directory.")

    def feature_engineering(self, df):
        """3. Feature Engineering: Create new features."""
        df = df.copy()
        
        # Create requested features
        df['Balance_to_Salary'] = df['Balance'] / (df['EstimatedSalary'] + 1e-6)
        df['Tenure_Age_Ratio'] = df['Tenure'] / df['Age']
        df['Products_Per_Tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
        df['IsHighValueCustomer'] = (df['Balance'] > 100000).astype(int)
        
        return df

    def preprocess_data(self):
        """2. Preprocessing & 4. Train-Test Split."""
        print("Engineering features...")
        self.df = self.feature_engineering(self.df)
        
        X = self.df.drop('Exited', axis=1)
        y = self.df['Exited']
        
        # Define columns for transformations
        categorical_cols = ['Geography', 'Gender']
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in categorical_cols]

        # 2. Preprocessing setup (OneHot for Geo, Scale for Numeric)
        # Using RobustScaler for scaling due to potential outliers in Balance/Salary
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, ['Geography']),
            ], remainder='passthrough')
        
        # Gender -> Label Encoding (since requested specifically)
        X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})
        
        # Stratified Split (80/20)
        print("Splitting and applying SMOTE...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Apply preprocessing
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names after transformation
        ohe_cols = self.preprocessor.named_transformers_['cat'].get_feature_names_out(['Geography'])
        self.feature_names = numeric_cols + list(ohe_cols) + ['Gender']
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
        
        return X_train_res, X_test_processed, y_train_res, y_test, X_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """5 & 6. Model Building & Hyperparameter Tuning, 7. Evaluation Metrics"""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        # Hyperparameter grids
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10]
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        best_roc_auc = 0
        best_model_obj = None
        results = []
        
        os.makedirs("model_outputs", exist_ok=True)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            # 6. RandomizedSearchCV to optimize for ROC-AUC
            search = RandomizedSearchCV(model, param_grids[name], n_iter=5, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            
            best_m = search.best_estimator_
            y_pred = best_m.predict(X_test)
            y_proba = best_m.predict_proba(X_test)[:, 1]
            
            # 7. Evaluation Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba)
            
            print(f"--- {name} Performance ---")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc:.4f}")
            
            results.append({
                'Model': name, 'Accuracy': acc, 'Precision': prec, 
                'Recall': rec, 'F1': f1, 'ROC-AUC': roc
            })
            
            # Save ROC Curve
            RocCurveDisplay.from_estimator(best_m, X_test, y_test)
            plt.title(f"{name} ROC Curve")
            plt.savefig(f"model_outputs/roc_curve_{name.replace(' ', '_')}.png")
            plt.close()
            
            # 9. Model Selection (Prioritize ROC-AUC & Recall)
            # We use a combined score to pick the best model automatically
            combined_score = roc + rec
            if combined_score > best_roc_auc:
                best_roc_auc = combined_score
                best_model_obj = best_m
                self.best_model_name = name

        print(f"\nBest Model Selected: {self.best_model_name}")
        self.model = best_model_obj
        
        # Save Evaluation Results
        pd.DataFrame(results).to_csv("model_outputs/model_comparison.csv", index=False)
        return X_test

    def explain_model(self, X_test_processed, X_test_raw):
        """8. Model Explainability: Feature Importance & SHAP"""
        print(f"\nGenerating Explanations for {self.best_model_name}...")
        
        if hasattr(self.model, 'feature_importances_'):
            # Feature Importance
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(X_test_processed.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_test_processed.shape[1]), [self.feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig("model_outputs/feature_importance.png")
            plt.close()
            
            # SHAP Values
            # Use a sample for SHAP to save compute time
            X_sample = pd.DataFrame(X_test_processed[:500], columns=self.feature_names)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig("model_outputs/shap_summary.png")
            plt.close()
            print("Explainability plots saved to 'model_outputs/'")

    def save_pipeline(self):
        """Save model and preprocessor for later use."""
        joblib.def_path = "churn_pipeline.pkl"
        pipeline_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }
        joblib.dump(pipeline_data, joblib.def_path)
        print(f"Pipeline saved successfully to {joblib.def_path}")

def predict_customer_churn(input_data: dict) -> float:
    """
    10. Output Predict Function
    Takes a dictionary of input data, engineers features, transforms, and predicts churn probability.
    """
    if not os.path.exists("churn_pipeline.pkl"):
        raise FileNotFoundError("Pipeline not found. Train the model first.")
        
    pipeline_data = joblib.load("churn_pipeline.pkl")
    model = pipeline_data['model']
    preprocessor = pipeline_data['preprocessor']
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # 1. Feature Engineering
    df['Balance_to_Salary'] = df['Balance'] / (df['EstimatedSalary'] + 1e-6)
    df['Tenure_Age_Ratio'] = df['Tenure'] / df['Age']
    df['Products_Per_Tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['IsHighValueCustomer'] = (df['Balance'] > 100000).astype(int)
    
    # Label Encode Gender
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    # Separate variables correctly before transformation
    categorical_cols = ['Geography']
    numeric_cols = df.drop(columns=['Geography', 'Gender']).columns.tolist()
    
    # Ensure column order matches preprocessor training
    # preprocessor expects: numeric_cols, then categorical
    processed_features = preprocessor.transform(df)
    
    # 2. Prediction
    probability = model.predict_proba(processed_features)[0, 1]
    return probability

if __name__ == "__main__":
    # Define dataset path (USER should place their CSV here or update the path)
    dataset_path = "European_Bank (2).csv" 
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please update the path.")
    else:
        pipeline = BankChurnPipeline(dataset_path)
        pipeline.load_data()
        pipeline.perform_eda()
        X_train_res, X_test_processed, y_train_res, y_test, X_test_raw = pipeline.preprocess_data()
        pipeline.train_and_evaluate(X_train_res, X_test_processed, y_train_res, y_test)
        pipeline.explain_model(X_test_processed, X_test_raw)
        pipeline.save_pipeline()
        
        print("\nPipeline execution complete! Try running app.py for the Streamlit UI.")
