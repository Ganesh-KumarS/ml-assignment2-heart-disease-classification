import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@st.cache_resource
def load_ml_models():
    try:
        models_list = {
            'Logistic Regression': joblib.load('model/logistic_regression.joblib'),
            'Decision Tree': joblib.load('model/decision_tree.joblib'),
            'KNN': joblib.load('model/knn.joblib'),
            'Naive Bayes': joblib.load('model/naive_bayes.joblib'),
            'SVM': joblib.load('model/svm.joblib')
        }
        sc = joblib.load('model/scaler.joblib')
        return models_list, sc
    except:
        return None, None

models, scaler = load_ml_models()

if models is None:
    st.error("Models not loaded")
    st.stop()

st.title("Heart Disease Classification System")
st.write("ML Project - Heart Disease Prediction using 5 Different Models")
st.write("---")

uploaded = st.file_uploader("Upload test data (CSV)", type=['csv'])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"Loaded data shape: {df.shape}")
    st.dataframe(df.head(), use_container_width=True)

    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        has_target = 'target' in df.columns

        model_choice = st.selectbox("Select model:", list(models.keys()))

        if st.button("Evaluate"):
            X = df[features]
            X_scaled = scaler.transform(X)

            m = models[model_choice]
            pred = m.predict(X_scaled)
            proba = m.predict_proba(X_scaled)

            if has_target:
                y_true = df['target']

                st.subheader("Evaluation Metrics")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Accuracy", f"{accuracy_score(y_true, pred):.4f}")
                c2.metric("AUC", f"{roc_auc_score(y_true, proba[:, 1]):.4f}")
                c3.metric("Precision", f"{precision_score(y_true, pred):.4f}")
                c4.metric("Recall", f"{recall_score(y_true, pred):.4f}")
                c5.metric("F1", f"{f1_score(y_true, pred):.4f}")
                c6.metric("MCC", f"{matthews_corrcoef(y_true, pred):.4f}")

                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_true, pred)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['No', 'Yes'],
                               yticklabels=['No', 'Yes'])
                    ax.set_ylabel('True')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)
                    plt.close()

                with c2:
                    st.write("**Classification Report**")
                    report = classification_report(y_true, pred,
                                                  target_names=['No Disease', 'Disease'],
                                                  output_dict=True)
                    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
            else:
                st.warning("No 'target' column found in uploaded data, cannot compute evaluation metrics.")
