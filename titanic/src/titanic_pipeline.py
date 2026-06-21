"""
Titanic Survival Prediction — End-to-End ML Pipeline
======================================================
Kaggle Competition: https://www.kaggle.com/competitions/titanic

This script walks through a complete ML workflow:
  1. Exploratory Data Analysis (EDA)
  2. Feature Engineering
  3. Preprocessing
  4. Model Training (Logistic Regression, Random Forest, XGBoost)
  5. Ensemble (Voting Classifier)
  6. Cross-Validation & Evaluation
  7. Submission file generation

Author: Hema Rani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Train shape: {train.shape}")  # (891 rows, 12 columns)
print(f"Test shape : {test.shape}")   # (418 rows, 11 columns) — no 'Survived'

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
# EDA = understanding your data BEFORE modeling.
# Goal: find patterns, spot missing values, and get feature ideas.

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

print("\n--- Survival Rate ---")
survival_rate = train["Survived"].mean()
print(f"Overall survival rate: {survival_rate:.2%}")
# ~38% survived — this is our baseline. A dumb model that predicts "died"
# for everyone would already be 62% accurate. We need to beat that.

print("\n--- Survival by Sex ---")
print(train.groupby("Sex")["Survived"].mean())
# Women survived at ~74%, men at ~19% — Sex is a VERY strong predictor.

print("\n--- Survival by Pclass ---")
print(train.groupby("Pclass")["Survived"].mean())
# 1st class: ~63%, 3rd class: ~24% — Class matters a lot.

print("\n--- Missing Values (Train) ---")
print(train.isnull().sum()[train.isnull().sum() > 0])
# Age: 177 missing (fill with median)
# Cabin: 687 missing (too sparse — extract deck letter or drop)
# Embarked: 2 missing (fill with mode)

print("\n--- Basic Stats ---")
print(train[["Age", "Fare", "SibSp", "Parch"]].describe())

# Save EDA plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Titanic EDA — Survival Patterns", fontsize=14, fontweight="bold")

# Plot 1: Survival by Sex
train.groupby("Sex")["Survived"].mean().plot(kind="bar", ax=axes[0, 0],
    color=["#e74c3c", "#3498db"], rot=0)
axes[0, 0].set_title("Survival Rate by Sex")
axes[0, 0].set_ylabel("Survival Rate")
axes[0, 0].set_ylim(0, 1)

# Plot 2: Survival by Pclass
train.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=axes[0, 1],
    color=["#2ecc71", "#f39c12", "#e74c3c"], rot=0)
axes[0, 1].set_title("Survival Rate by Pclass")
axes[0, 1].set_ylabel("Survival Rate")
axes[0, 1].set_ylim(0, 1)

# Plot 3: Age distribution by survival
train[train["Survived"] == 1]["Age"].dropna().hist(ax=axes[1, 0], alpha=0.7,
    color="#2ecc71", label="Survived", bins=20)
train[train["Survived"] == 0]["Age"].dropna().hist(ax=axes[1, 0], alpha=0.7,
    color="#e74c3c", label="Died", bins=20)
axes[1, 0].set_title("Age Distribution by Survival")
axes[1, 0].legend()

# Plot 4: Fare by survival
train.boxplot(column="Fare", by="Survived", ax=axes[1, 1])
axes[1, 1].set_title("Fare by Survival")
axes[1, 1].set_xlabel("Survived (0=No, 1=Yes)")
plt.suptitle("")  # Remove auto-generated title from boxplot

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_plots.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nEDA plots saved to outputs/eda_plots.png")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Feature engineering = creating new meaningful inputs from existing raw data.
# Raw data often has hidden signals — our job is to extract them.

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

def engineer_features(df):
    df = df.copy()

    # --- Title from Name ---
    # Names like "Mr.", "Mrs.", "Miss.", "Master." carry survival signal.
    # "Master" = young boys; "Miss" = unmarried women (often young).
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    # Rare titles → group into "Rare" to avoid overfitting on tiny groups
    rare_titles = ["Capt", "Col", "Countess", "Don", "Dona", "Dr",
                   "Jonkheer", "Lady", "Major", "Rev", "Sir"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # --- Family Size ---
    # SibSp = siblings/spouses aboard; Parch = parents/children aboard
    # Alone passengers and very large families had lower survival rates.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # +1 for self
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # --- Cabin Deck ---
    # Cabin number encodes which deck (A-G). Higher decks = closer to lifeboats.
    # 77% of values are missing → extract letter or mark as "Unknown"
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    # --- Fare Bin ---
    # Instead of raw Fare (continuous), bin into quartiles.
    # This makes the model less sensitive to outliers (someone paying $512).
    df["FareBin"] = pd.qcut(df["Fare"].fillna(df["Fare"].median()), 4, labels=False)

    # --- Age Bin ---
    df["AgeBin"] = pd.cut(df["Age"].fillna(df["Age"].median()),
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])

    return df

train = engineer_features(train)
test  = engineer_features(test)

print("New features created: Title, FamilySize, IsAlone, CabinKnown, Deck, FareBin, AgeBin")
print(f"\nTitle value counts:\n{train['Title'].value_counts()}")
print(f"\nFamilySize distribution:\n{train['FamilySize'].value_counts().sort_index()}")

# ─────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────
# Preprocessing = converting raw/engineered data into numbers the model can use.
# ML models work with numbers only — no strings, no NaNs.

print("\n" + "=" * 60)
print("STEP 4: Preprocessing")
print("=" * 60)

def preprocess(df, is_train=True):
    df = df.copy()

    # Fill missing values
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical → numbers
    # Label encoding: maps each category to an integer
    df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Title"]    = df["Title"].map(
        {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}).fillna(4)
    df["AgeBin"]   = df["AgeBin"].map(
        {"Child": 0, "Teen": 1, "YoungAdult": 2, "Adult": 3, "Senior": 4}).fillna(2)
    df["Deck"]     = df["Deck"].map(
        {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "Unknown": 7}).fillna(7)

    # Select final features for the model
    features = [
        "Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked",
        "Title", "FamilySize", "IsAlone", "CabinKnown", "FareBin", "AgeBin", "Deck"
    ]

    X = df[features]
    return X

X_train_raw = preprocess(train)
X_test_raw  = preprocess(test)
y_train     = train["Survived"]

# Feature Scaling — important for Logistic Regression (not needed for trees, but harmless)
# StandardScaler: makes each feature have mean=0, std=1
# Why? LR uses gradient descent — unscaled features can make it converge slowly/poorly.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)   # fit on train (learn mean/std), then transform
X_test  = scaler.transform(X_test_raw)        # ONLY transform test (use train's mean/std!)
# ^ This is a crucial concept: never fit the scaler on test data. That would be data leakage.

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape    : {X_test.shape}")
print(f"\nFeatures used: {list(X_train_raw.columns)}")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING & CROSS-VALIDATION
# ─────────────────────────────────────────────
# We train 3 models and compare them using cross-validation.
# Cross-validation: splits training data into k folds, trains on k-1, tests on 1.
# This gives a reliable estimate of how the model performs on unseen data.

print("\n" + "=" * 60)
print("STEP 5: Model Training & Cross-Validation")
print("=" * 60)

# StratifiedKFold: ensures each fold has the same class ratio (important for imbalanced data)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Model 1: Logistic Regression ---
# Simple linear model. Good baseline. Fast, interpretable.
# Works well when features have a roughly linear relationship with the target.
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="accuracy")
print(f"\nLogistic Regression  — CV Accuracy: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")

# --- Model 2: Random Forest ---
# Ensemble of decision trees. More powerful, handles non-linear relationships.
# n_estimators = number of trees; more trees = more stable, but slower.
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=4,
                             random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")
print(f"Random Forest        — CV Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

# --- Model 3: XGBoost ---
# Gradient boosting: trees built sequentially, each correcting the previous one's errors.
# Usually the strongest performer on tabular data.
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     eval_metric="logloss", random_state=42, verbosity=0)
xgb_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="accuracy")
print(f"XGBoost              — CV Accuracy: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")

# --- Model 4: Voting Ensemble ---
# Combines all 3 models — each votes, majority wins.
# Often beats individual models by reducing variance (averaging out each model's mistakes).
ensemble = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
    voting="soft"  # "soft" uses predicted probabilities (more nuanced than hard votes)
)
ens_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring="accuracy")
print(f"Voting Ensemble      — CV Accuracy: {ens_scores.mean():.4f} ± {ens_scores.std():.4f}")

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (Random Forest)")
print("=" * 60)

rf.fit(X_train, y_train)
feature_names = list(X_train_raw.columns)
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(importances.to_string())

# Plot feature importances
plt.figure(figsize=(10, 5))
importances.plot(kind="bar", color="#3498db")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nFeature importance plot saved to outputs/feature_importance.png")

# ─────────────────────────────────────────────
# 7. TRAIN FINAL MODEL & GENERATE SUBMISSION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Final Model Training & Submission")
print("=" * 60)

# Train the ensemble on ALL training data (no validation split this time)
ensemble.fit(X_train, y_train)

# Predict on test set
test_predictions = ensemble.predict(X_test)

# Build submission file
# Kaggle expects exactly: PassengerId, Survived
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved: {submission_path}")
print(f"Predicted survivors: {test_predictions.sum()} / {len(test_predictions)}")
print(f"Predicted survival rate: {test_predictions.mean():.2%}")

# Quick sanity check on training data
train_preds = ensemble.predict(X_train)
print(f"\nTraining accuracy (in-sample): {(train_preds == y_train).mean():.4f}")
print("Note: in-sample accuracy is always higher — use CV scores for real performance estimates.")

print("\n" + "=" * 60)
print("DONE! Check outputs/ folder for:")
print("  - submission.csv   → upload to Kaggle")
print("  - eda_plots.png    → survival pattern visualizations")
print("  - feature_importance.png → which features matter most")
print("=" * 60)
