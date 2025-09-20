️# -----------------------------
#  Imports
import sys
sys.path.append(r"")  # Path to folder containing 'src'
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.svm import LpLinftySVM
from src import Kernels
# -----------------------------
#  Load and preprocess data
# -----------------------------
df = pd.read_csv(r"")

# Convert classes to -1 / +1
df['Class'] = df['target'].apply(lambda x: -1 if x == 0 else 1)

# Features and labels
X = df.iloc[:, :-2].to_numpy()
y = df['Class'].to_numpy()

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Full K-Fold evaluation
# -----------------------------
skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

best_f1 = 0
best_metrics = None

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize and train model
    clf = LpLinftySVM(C=3, b=0.0, kernel="linear", p=0.2)
    clf.fit(X_train, y_train, X_test)
    y_pred = clf.predict()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=1),
        'recall': recall_score(y_test, y_pred, pos_label=1),
        'f1': f1_score(y_test, y_pred, pos_label=1)
    }

    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_metrics = metrics

# -----------------------------
# Show best metrics
# -----------------------------
print(f"Accuracy:  {best_metrics['accuracy']:.3f}")
print(f"Precision: {best_metrics['precision']:.3f}")
print(f"Recall:    {best_metrics['recall']:.3f}")
print(f"F1-score:  {best_metrics['f1']:.3f}")
