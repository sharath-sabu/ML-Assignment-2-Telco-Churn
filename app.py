import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("📊 Telco Customer Churn Prediction")
st.write("Machine Learning Models for Predicting Customer Churn")

# -------------------------------
# Load Models
# -------------------------------
BASE_PATH = "model"

models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_PATH, "logistic.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_PATH, "decision_tree.pkl")),
    "kNN": joblib.load(os.path.join(BASE_PATH, "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(BASE_PATH, "naive_bayes.pkl")),
    "Random Forest (Ensemble)": joblib.load(os.path.join(BASE_PATH, "random_forest.pkl")),
    "XGBoost (Ensemble)": joblib.load(os.path.join(BASE_PATH, "xgboost.pkl")),
}

scaler = joblib.load(os.path.join(BASE_PATH, "scaler.pkl"))

# -------------------------------
# Sidebar - Model Selection
# -------------------------------
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a Model", list(models.keys()))
model = models[selected_model_name]

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    if "Churn Value" not in data.columns:
        st.error("Uploaded file must contain 'Churn Value' column for evaluation.")
    else:
        X = data.drop("Churn Value", axis=1)
        y = data["Churn Value"]

        # Ensure Total Charges numeric
        if "Total Charges" in X.columns:
            X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")
            X["Total Charges"] = X["Total Charges"].fillna(X["Total Charges"].median())

        # One-hot encoding
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Align columns with training features
        training_features = model.feature_names_in_
        X_encoded = X_encoded.reindex(columns=training_features, fill_value=0)

        # Scaling for specific models
        if selected_model_name in ["Logistic Regression", "kNN"]:
            X_encoded = scaler.transform(X_encoded)

        # Predictions
        y_pred = model.predict(X_encoded)
        y_prob = model.predict_proba(X_encoded)[:, 1]

        # Metrics
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(acc, 4))
        col2.metric("AUC", round(auc, 4))
        col3.metric("Precision", round(precision, 4))

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", round(recall, 4))
        col5.metric("F1 Score", round(f1, 4))
        col6.metric("MCC", round(mcc, 4))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
