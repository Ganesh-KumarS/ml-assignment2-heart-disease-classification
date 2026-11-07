import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Heart disease dataset from UCI repo
# 4 sites combined: Cleveland, Hungarian, Switzerland, VA Long Beach
base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"

sites = {
    'cleveland': 'processed.cleveland.data',
    'hungarian': 'processed.hungarian.data',
    'switzerland': 'processed.switzerland.data',
    'va_long_beach': 'processed.va.data',
}

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

print("Loading data from UCI...")
dfs = []
for site_name, file_name in sites.items():
    df_site = pd.read_csv(base_url + file_name, header=None, names=cols, na_values='?')
    print(f"  {site_name}: {len(df_site)} rows")
    dfs.append(df_site)

# Combine all sites
data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(data)}")

# Convert target to binary (0=no, 1=yes)
data['target'] = (data['target'] > 0).astype(int)

# Fill missing values
print(f"\nMissing before: {data.isnull().sum().sum()}")

numeric_cols = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']
cat_cols = ['ca', 'thal']

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print(f"Missing after: {data.isnull().sum().sum()}")

print(f"\nTarget distribution:")
print(data['target'].value_counts())

# Save full dataset
data.to_csv('heart_disease_full.csv', index=False)
print("\nSaved heart_disease_full.csv")

# Create test split (must match the train/test split used in train_models.py
# so that test_data.csv is a true held-out set with zero overlap with the
# data the saved models were trained on)
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['target'])
test.to_csv('test_data.csv', index=False)
print(f"Saved test_data.csv ({len(test)} rows)")
