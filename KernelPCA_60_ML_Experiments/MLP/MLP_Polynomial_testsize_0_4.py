"""
MLP with Kernel PCA (Polynomial kernel)
Train/test split: 0.6/0.4
Generated as one of 60 independent experiments.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.neural_network import MLPClassifier


# 1. Load the three dataset files.
df1 = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=True)
df2 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv", low_memory=True)
df3 = pd.read_csv("Thursday-workingHours-Morning-WebAttacks.pcap_ISCX.csv", low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()
dataset = dataset.dropna(subset=[dataset.columns[-1]])

# 2. Separate features and target.
X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1].astype(str).copy()

X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. Encode target labels and impute missing feature values.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Fit imputer on the training data only to avoid preprocessing leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=0,
    stratify=y
)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 4. Sample rows to keep Kernel PCA computationally manageable.
# Features and labels use matching rows so their lengths remain consistent.
train_count = min(10000, len(X_train))
test_count = min(2000, len(X_test))

X_train_sample = X_train[:train_count]
y_train_sample = y_train[:train_count]
X_test_sample = X_test[:test_count]
y_test_sample = y_test[:test_count]

# 5. Apply Kernel PCA.
kpca = KernelPCA(
    n_components=2, kernel="poly", gamma=15, degree=3, coef0=1
)
X_train_kpca = kpca.fit_transform(X_train_sample)
X_test_kpca = kpca.transform(X_test_sample)

print("=" * 65)
print("MLP | Kernel PCA: Polynomial | test_size=0.4")
print("Train shape:", X_train_kpca.shape)
print("Test shape:", X_test_kpca.shape)
print("=" * 65)

# 6. Train the model and predict.
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42,
    early_stopping=True
)

model.fit(X_train_kpca, y_train_sample)
y_pred = model.predict(X_test_kpca)

# 7. Evaluate.
all_labels = np.arange(len(label_encoder.classes_))
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test_sample, y_pred, labels=all_labels))

print("\\nAccuracy:", accuracy_score(y_test_sample, y_pred))
print("Weighted Precision:", precision_score(
    y_test_sample, y_pred, average="weighted", zero_division=0
))
print("Weighted Recall:", recall_score(
    y_test_sample, y_pred, average="weighted", zero_division=0
))
print("Weighted F1 Score:", f1_score(
    y_test_sample, y_pred, average="weighted", zero_division=0
))

print("\\nClassification Report:")
print(classification_report(
    y_test_sample,
    y_pred,
    labels=all_labels,
    target_names=label_encoder.classes_,
    zero_division=0
))
