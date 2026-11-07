# Heart Disease Classification - ML Project

## Problem Statement

This project aims to build machine learning models to predict heart disease using medical patient data. The goal is to compare 5 different classification algorithms and evaluate which one performs best.

## Dataset

**Source**: UCI Heart Disease Dataset (combined from 4 sites)
- Cleveland, Hungarian, Switzerland, VA Long Beach

**Dataset Info:**
- Total samples: 920 patient records
- Number of features: 13 medical attributes
- Target: Binary classification (0 = No disease, 1 = Disease present)

**Features Used:**
- age - Patient age in years
- sex - Gender (1=M, 0=F)
- cp - Chest pain type (0-3)
- trestbps - Resting blood pressure
- chol - Serum cholesterol level
- fbs - Fasting blood sugar
- restecg - Resting ECG results
- thalach - Maximum heart rate
- exang - Exercise induced angina
- oldpeak - ST depression
- slope - ST segment slope
- ca - Coronary artery vessels
- thal - Thalassemia type

Data was combined from multiple UCI sites and missing values were imputed with median/mode to meet the 500+ instance requirement.

## Models Implemented

1. **Logistic Regression** - Linear classification model
2. **Decision Tree** - Tree-based classifier
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes** - Probabilistic classifier
5. **Support Vector Machine (SVM)** - Kernel-based classifier

## Evaluation Metrics

For each model, we calculated:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- MCC (Matthews Correlation Coefficient)

## Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8207 | 0.8959 | 0.8224 | 0.8627 | 0.8421 | 0.6358 |
| Decision Tree | 0.7826 | 0.7949 | 0.7925 | 0.8235 | 0.8077 | 0.5585 |
| KNN | 0.8315 | 0.8690 | 0.8257 | 0.8824 | 0.8531 | 0.6581 |
| Naive Bayes | 0.8424 | 0.8864 | 0.8614 | 0.8529 | 0.8571 | 0.6814 |
| **SVM** | **0.8478** | **0.9094** | 0.8246 | **0.9216** | **0.8704** | **0.6938** |

**Best Model: SVM** with highest accuracy (84.78%), AUC, recall, and F1 score.

## Observations

| Model | Observation |
|-------|-------------|
| Logistic Regression | Good baseline with balanced performance; fast to train and predict; assumes linear decision boundaries. |
| Decision Tree | Easy to interpret; lowest accuracy among all models; prone to overfitting. |
| KNN | Good performance (83.15% accuracy); captures non-linear patterns well; computationally expensive for large datasets. |
| Naive Bayes | Fast training and prediction; highest precision (86.14%); assumes feature independence. |
| SVM | Best overall performer; handles complex decision boundaries well; takes longer to train. |
| **Overall Winner** | **SVM** — highest accuracy, AUC, recall, F1, and MCC among all 5 models. |

## Project Structure

```
├── app.py                  # Streamlit web application
├── train_models.py         # Training script for all 5 models
├── data_preparation.py     # Data loading and preprocessing
├── requirements.txt        # Python dependencies
├── heart_disease_full.csv  # Full dataset
├── test_data.csv          # Test dataset for evaluation
└── model/                 # Saved trained models
    ├── scaler.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare data:
   ```
   python data_preparation.py
   ```

3. Train models:
   ```
   python train_models.py
   ```

4. Run the web app:
   ```
   streamlit run app.py
   ```

## Deployment

The application is deployed on Streamlit Community Cloud: `<add your live Streamlit app URL here after deploying>`

## GitHub Repository

https://github.com/Ganesh-KumarS/ml-assignment2-heart-disease-classification

