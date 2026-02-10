"""
Step-by-step semiconductor process failure prediction workflow.

This script demonstrates a complete, reproducible classification workflow with
clear explanations and intermediate checks across four phases:
1) Data setup and exploration
2) Cleaning and feature engineering
3) Modeling and evaluation
4) Insights and operational recommendations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# -----------------------------------------------------------------------------
# Phase 1: Data Setup and Exploration
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 1: DATA SETUP AND EXPLORATION")
print("=" * 80)

# We create a synthetic dataset to mimic process sensor readings and failure labels.
# - n_informative features carry signal related to failures.
# - class_sep controls separability.
# - weights creates intentional class imbalance, common in failure prediction.
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_clusters_per_class=2,
    weights=[0.88, 0.12],
    class_sep=1.2,
    random_state=42,
)

feature_names = [f"sensor_{i:02d}" for i in range(1, 21)]
df = pd.DataFrame(X, columns=feature_names)
df["failure_label"] = y

# Add intentionally "unusable" columns to illustrate dropping logic.
# - lot_id: unique identifier (generally non-predictive leakage-prone ID)
# - notes: mostly missing text field
rng = np.random.default_rng(42)
df["lot_id"] = np.arange(100_001, 100_001 + len(df))
df["notes"] = pd.Series(np.where(rng.random(len(df)) < 0.95, None, "manual_check"), dtype="object")

# Inject missing values into numeric columns to simulate sensor gaps.
missing_rate = 0.06
for col in feature_names:
    missing_mask = rng.random(len(df)) < missing_rate
    df.loc[missing_mask, col] = np.nan

print("\n1) Dataset shape:")
print(df.shape)

print("\n2) Data types:")
print(df.dtypes)

print("\n3) Missing values per column:")
missing_summary = df.isna().sum().sort_values(ascending=False)
print(missing_summary)

# Class imbalance check against predefined threshold.
# Here, we require the minority class to be at least 10%.
imbalance_threshold = 0.10
class_distribution = df["failure_label"].value_counts(normalize=True).sort_index()
minority_ratio = class_distribution.min()
passes_threshold = minority_ratio >= imbalance_threshold

print("\n4) Class distribution (normalized):")
print(class_distribution)
print(
    f"Minority ratio = {minority_ratio:.3f}; "
    f"threshold = {imbalance_threshold:.2f}; "
    f"passes_threshold = {passes_threshold}"
)

# Basic distribution summaries and visualizations.
print("\n5) Quick descriptive statistics for numeric features:")
print(df[feature_names].describe().T[["mean", "std", "min", "max"]].head(10))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# A key feature histogram.
df["sensor_01"].hist(ax=axes[0], bins=30)
axes[0].set_title("Distribution: sensor_01")
axes[0].set_xlabel("sensor_01")

# Boxplot for another feature to inspect spread/outliers.
axes[1].boxplot(df["sensor_02"].dropna(), vert=True)
axes[1].set_title("Boxplot: sensor_02")
axes[1].set_ylabel("sensor_02")

# Target class balance bar chart.
class_counts = df["failure_label"].value_counts().sort_index()
axes[2].bar(class_counts.index.astype(str), class_counts.values)
axes[2].set_title("Target Class Counts")
axes[2].set_xlabel("failure_label")
axes[2].set_ylabel("count")

plt.tight_layout()
plt.savefig("phase1_distributions.png", dpi=140)
plt.close()
print("Saved visualization: phase1_distributions.png")


# -----------------------------------------------------------------------------
# Phase 2: Cleaning and Feature Engineering
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 2: CLEANING AND FEATURE ENGINEERING")
print("=" * 80)

# Drop unusable/irrelevant columns:
# - lot_id is an identifier and should not be used as a predictive signal.
# - notes has very high missingness and sparse text content in this synthetic setup.
columns_to_drop = ["lot_id", "notes"]
model_df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# Separate features and target.
X_full = model_df.drop(columns=["failure_label"])
y_full = model_df["failure_label"]

# Train-test split ratio: 80% train, 20% test using stratification to preserve class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=0.20,
    random_state=42,
    stratify=y_full,
)
print(
    f"Train-test split complete: X_train={X_train.shape}, X_test={X_test.shape}, "
    "ratio=80:20 (stratified)."
)

# Preprocessing strategy:
# - Numeric imputation with median (robust to skew/outliers).
# - Standardization for scale-sensitive models (LogReg, SVM).
numeric_features = X_full.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
)

# Baseline model for first comparison: Logistic Regression.
baseline_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ]
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_proba = baseline_model.predict_proba(X_test)[:, 1]

baseline_f1 = f1_score(y_test, baseline_pred)
baseline_auc = roc_auc_score(y_test, baseline_proba)
print(f"Baseline (Logistic Regression) F1={baseline_f1:.3f}, ROC-AUC={baseline_auc:.3f}")


# -----------------------------------------------------------------------------
# Phase 3: Modeling and Evaluation
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 3: MODELING AND EVALUATION")
print("=" * 80)

# Define three classification models.
models = {
    "LogisticRegression": LogisticRegression(max_iter=3000, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=350,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []
fitted_pipelines = {}

for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, proba)

    results.append(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }
    )
    fitted_pipelines[model_name] = pipeline

# Pretty print results.
results_df = pd.DataFrame(results).drop(columns=["confusion_matrix"])
results_df = results_df.sort_values(by=["f1", "roc_auc"], ascending=False)

print("\nModel comparison (sorted by F1, then ROC-AUC):")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nConfusion matrices:")
for row in results:
    print(f"\n{row['model']} confusion matrix:")
    print(row["confusion_matrix"])

best_model_name = results_df.iloc[0]["model"]
print(f"\nBest performing model by F1/ROC-AUC tie-break: {best_model_name}")
print(
    "Interpretation: the best model likely balances false positives and false negatives "
    "better for this imbalanced failure-detection scenario."
)


# -----------------------------------------------------------------------------
# Phase 4: Insights and Final Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 4: INSIGHTS AND FINAL SUMMARY")
print("=" * 80)

# Feature importance analysis.
# Prefer direct importances from Random Forest when available; otherwise fall back to
# permutation importance on the selected best model.
if "RandomForest" in fitted_pipelines:
    rf_pipe = fitted_pipelines["RandomForest"]
    rf_clf = rf_pipe.named_steps["classifier"]
    importances = pd.Series(rf_clf.feature_importances_, index=numeric_features)
    importance_method = "Random Forest built-in feature_importances_"
else:
    best_pipe = fitted_pipelines[best_model_name]
    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=8, random_state=42)
    importances = pd.Series(perm.importances_mean, index=numeric_features)
    importance_method = "Permutation importance"

importance_rank = importances.sort_values(ascending=False)
top_k = 8
top_features = importance_rank.head(top_k)

print(f"Feature importance method: {importance_method}")
print(f"Top {top_k} most influential variables:")
for feature, score in top_features.items():
    print(f"  - {feature}: {score:.4f}")

# Operational recommendation: monitor top features most linked to failure risk.
recommended_monitoring = top_features.index.tolist()[:5]
print("\nOperational monitoring recommendation (top 5):")
for feat in recommended_monitoring:
    print(f"  - Closely track {feat} in SPC/control charts and alerting rules.")

# Save importance chart.
plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind="barh")
plt.title("Top Feature Importances for Failure Prediction")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("phase4_feature_importance.png", dpi=140)
plt.close()
print("Saved visualization: phase4_feature_importance.png")

print("\nWorkflow complete.")
