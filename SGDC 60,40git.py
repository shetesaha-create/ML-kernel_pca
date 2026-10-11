```python
import numpy as np
import pandas as pd

# =========================
# 1. LOAD DATA
# =========================

df1 = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# =========================
# 2. PREPROCESSING
# =========================

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# =========================
# 3. TRAIN TEST SPLIT
# =========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.6,
    random_state=0,
    stratify=y
)

# =========================
# 4. KERNEL PCA
# =========================

from sklearn.decomposition import KernelPCA

X_train_sample = X_train[:10000]
X_test_sample = X_test[:2000]
y_train_sample = y_train[:10000]
y_test_sample = y_test[:2000]

kpca = KernelPCA(
    n_components=2,
    kernel='cosine'
)

X_train_kpca = kpca.fit_transform(X_train_sample)
X_test_kpca = kpca.transform(X_test_sample)

# =========================
# 5. MULTILAYER PERCEPTRON
# =========================

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)

classifier.fit(X_train_kpca, y_train_sample)
y_pred = classifier.predict(X_test_kpca)

# =========================
# 6. EVALUATION
# =========================

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Cosine Kernel + Test Size 0.6")

print("Confusion Matrix:\n")
print(confusion_matrix(y_test_sample, y_pred))

print("\nAccuracy : {:.4f}".format(
    accuracy_score(y_test_sample, y_pred)
))

print("\nPrecision : {:.4f}".format(
    precision_score(
        y_test_sample, y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nRecall : {:.4f}".format(
    recall_score(
        y_test_sample, y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nF1 Score : {:.4f}".format(
    f1_score(
        y_test_sample, y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nClassification Report:\n")
print(classification_report(
    y_test_sample, y_pred,
    zero_division=0
))
```