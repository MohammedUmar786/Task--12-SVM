# ============================================
# TASK 12: SUPPORT VECTOR MACHINES (SVM)
# ============================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# ============================================
# Step 2: Load Dataset
# ============================================

dataset = pd.read_csv("Social_Network_Ads.csv")

print("First 5 Rows:")
print(dataset.head())

print("\nDataset Shape:")
print(dataset.shape)

print("\nDataset Info:")
print(dataset.info())

# ============================================
# Step 3: Select Features and Target
# ============================================

# Features: Age and EstimatedSalary
X = dataset.iloc[:, [2, 3]].values

# Target: Purchased
y = dataset.iloc[:, 4].values

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================
# Step 4: Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ============================================
# Step 5: Feature Scaling
# ============================================

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ============================================
# Step 6: Train SVM Models
# ============================================

# -------------------------------
# Linear Kernel
# -------------------------------
linear_svm = SVC(kernel='linear', random_state=0)
linear_svm.fit(X_train, y_train)

# -------------------------------
# Polynomial Kernel
# -------------------------------
poly_svm = SVC(kernel='poly', degree=3, random_state=0)
poly_svm.fit(X_train,# Task--12-SVM
