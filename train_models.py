import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib

print("Loading data...")
df = pd.read_csv('heart_disease_full.csv')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'model/scaler.joblib')

print(f"Train shape: {X_train_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")

results = {}

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, lr_pred),
    'AUC': roc_auc_score(y_test, lr_proba),
    'Precision': precision_score(y_test, lr_pred),
    'Recall': recall_score(y_test, lr_pred),
    'F1': f1_score(y_test, lr_pred),
    'MCC': matthews_corrcoef(y_test, lr_pred)
}
joblib.dump(lr_model, 'model/logistic_regression.joblib')
print(f"Accuracy: {results['Logistic Regression']['Accuracy']:.4f}")

# Train Decision Tree
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_proba = dt_model.predict_proba(X_test_scaled)[:, 1]

results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, dt_pred),
    'AUC': roc_auc_score(y_test, dt_proba),
    'Precision': precision_score(y_test, dt_pred),
    'Recall': recall_score(y_test, dt_pred),
    'F1': f1_score(y_test, dt_pred),
    'MCC': matthews_corrcoef(y_test, dt_pred)
}
joblib.dump(dt_model, 'model/decision_tree.joblib')
print(f"Accuracy: {results['Decision Tree']['Accuracy']:.4f}")

# Train KNN
print("\nTraining KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

results['KNN'] = {
    'Accuracy': accuracy_score(y_test, knn_pred),
    'AUC': roc_auc_score(y_test, knn_proba),
    'Precision': precision_score(y_test, knn_pred),
    'Recall': recall_score(y_test, knn_pred),
    'F1': f1_score(y_test, knn_pred),
    'MCC': matthews_corrcoef(y_test, knn_pred)
}
joblib.dump(knn_model, 'model/knn.joblib')
print(f"Accuracy: {results['KNN']['Accuracy']:.4f}")

# Train Naive Bayes
print("\nTraining Naive Bayes...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_pred = nb_model.predict(X_test_scaled)
nb_proba = nb_model.predict_proba(X_test_scaled)[:, 1]

results['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test, nb_pred),
    'AUC': roc_auc_score(y_test, nb_proba),
    'Precision': precision_score(y_test, nb_pred),
    'Recall': recall_score(y_test, nb_pred),
    'F1': f1_score(y_test, nb_pred),
    'MCC': matthews_corrcoef(y_test, nb_pred)
}
joblib.dump(nb_model, 'model/naive_bayes.joblib')
print(f"Accuracy: {results['Naive Bayes']['Accuracy']:.4f}")

# Train SVM
print("\nTraining SVM...")
svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

results['SVM'] = {
    'Accuracy': accuracy_score(y_test, svm_pred),
    'AUC': roc_auc_score(y_test, svm_proba),
    'Precision': precision_score(y_test, svm_pred),
    'Recall': recall_score(y_test, svm_pred),
    'F1': f1_score(y_test, svm_pred),
    'MCC': matthews_corrcoef(y_test, svm_pred)
}
joblib.dump(svm_model, 'model/svm.joblib')
print(f"Accuracy: {results['SVM']['Accuracy']:.4f}")

# Save results
results_df = pd.DataFrame(results).T
print("\nResults summary:")
print(results_df.round(4))

results_df.to_csv('model_results.csv')
print("\nResults saved to model_results.csv")

# Print best models
print("\nBest models by metric:")
for col in results_df.columns:
    best_model = results_df[col].idxmax()
    print(f"  {col}: {best_model} ({results_df.loc[best_model, col]:.4f})")
