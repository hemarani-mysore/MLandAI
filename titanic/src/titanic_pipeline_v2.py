"""
Titanic Survival Prediction — End-to-End ML Pipeline v2
========================================================
Kaggle Competition: https://www.kaggle.com/competitions/titanic

Improvements over v1:
  - Age filled by Title median (not overall median)
  - FarePerPerson feature (fare divided by family size)
  - Family size bucketed (Alone / Small / Large)
  - Age x Pclass interaction feature
  - Tighter regularization to reduce overfitting

CV Accuracy: ~84.5% (XGBoost), ~84% (Ensemble)

Author: Hema Rani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS — update BASE_DIR to your local path
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
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

print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

def extract_title(df):
    """Extract title from Name column and group rare titles."""
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare = ['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Rev','Sir']
    df['Title'] = df['Title'].replace(rare, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    return df

train = extract_title(train)
test  = extract_title(test)

# FIX 1: Fill Age by Title group median
# Why? A "Master" (young boy) has median age 3.5, not 28 (overall median).
# Using group median is far more accurate than a single global value.
age_medians = train.groupby('Title')['Age'].median()
print("Age median by Title:")
print(age_medians.to_string())

for df in [train, test]:
    for title, median_age in age_medians.items():
        mask = (df['Title'] == title) & (df['Age'].isnull())
        df.loc[mask, 'Age'] = median_age
    df['Age'] = df['Age'].fillna(train['Age'].median())  # fallback

def engineer_features(df, fare_median):
    df = df.copy()

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # FIX 2: Bucket family size — cleaner signal than raw number
    # Alone: worst odds (0.30), Small (2-4): best odds (0.58), Large (5+): bad (0.16)
    df['FamBucket'] = pd.cut(df['FamilySize'], bins=[0,1,4,20],
                              labels=[0,1,2]).astype(int)
    df['IsAlone']   = (df['FamilySize'] == 1).astype(int)

    # Cabin features
    df['CabinKnown'] = df['Cabin'].notna().astype(int)
    df['Deck']       = df['Cabin'].str[0].fillna('Unknown')

    # FIX 3: Fare per person
    # A group of 4 sharing a $100 ticket each effectively paid $25.
    # Raw fare inflates the signal for larger families.
    df['Fare']          = df['Fare'].fillna(fare_median)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FareBin']       = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')

    # FIX 4: Age x Pclass interaction
    # Young passengers in 3rd class had very different odds than young in 1st class.
    # Multiplying captures this combined effect.
    df['Age_Pclass'] = df['Age'] * df['Pclass']

    return df

fare_median = train['Fare'].median()
train = engineer_features(train, fare_median)
test  = engineer_features(test, fare_median)

print(f"\nFeatures created: FamilySize, FamBucket, IsAlone, CabinKnown, Deck, "
      f"FarePerPerson, FareBin, Age_Pclass")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

def preprocess(df, fare_median):
    df = df.copy()
    df['Embarked']      = df['Embarked'].fillna('S')
    df['FarePerPerson'] = df['FarePerPerson'].fillna(fare_median)

    # Encode categoricals to integers
    df['Sex']      = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
    df['Title']    = df['Title'].map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}).fillna(4)
    df['Deck']     = df['Deck'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'Unknown':7}).fillna(7)

    features = [
        'Pclass', 'Sex', 'Age', 'Fare', 'FarePerPerson',
        'SibSp', 'Parch', 'Embarked', 'Title', 'FamilySize',
        'FamBucket', 'IsAlone', 'CabinKnown', 'FareBin', 'Deck', 'Age_Pclass'
    ]
    X = df[features].fillna(df[features].median())  # safety net for any remaining NaNs
    return X

X_train_raw = preprocess(train, fare_median)
X_test_raw  = preprocess(test, fare_median)
y_train     = train['Survived']

# Verify no NaNs before modeling
assert X_train_raw.isnull().sum().sum() == 0, "NaNs found in training features!"
assert X_test_raw.isnull().sum().sum()  == 0, "NaNs found in test features!"

# Scale features — important for Logistic Regression
# fit_transform on train (learns mean/std), transform only on test (no leakage!)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Training features: {X_train.shape}")
print(f"Test features    : {X_test.shape}")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING & CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Training & Cross-Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression — C=0.1 means stronger regularization (less overfitting)
lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)

# Random Forest — shallower trees (max_depth=5) to reduce overfitting
rf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=5,
                             max_features='sqrt', random_state=42, n_jobs=-1)

# XGBoost — lower learning_rate + more trees = slower but better generalization
xgb = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                     eval_metric='logloss', random_state=42, verbosity=0)

print("\nCross-validation scores (5-fold):")
for name, model in [('Logistic Regression', lr), ('Random Forest', rf), ('XGBoost', xgb)]:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  {name:22s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Voting Ensemble — combines all 3 with soft voting (uses probabilities)
ensemble   = VotingClassifier([('lr',lr),('rf',rf),('xgb',xgb)], voting='soft')
ens_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
print(f"  {'Voting Ensemble':22s}: {ens_scores.mean():.4f} ± {ens_scores.std():.4f}")

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature Importance")
print("=" * 60)

rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_,
                        index=X_train_raw.columns).sort_values(ascending=False)
print(importances.to_string())

plt.figure(figsize=(10, 5))
importances.plot(kind='bar', color='#3498db')
plt.title('Feature Importance (Random Forest) — v2')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_v2.png'), dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 6. GENERATE SUBMISSION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Final Model & Submission")
print("=" * 60)

ensemble.fit(X_train, y_train)
preds = ensemble.predict(X_test)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds})
sub_path   = os.path.join(OUTPUT_DIR, 'submission_v2.csv')
submission.to_csv(sub_path, index=False)

print(f"Submission saved: {sub_path}")
print(f"Predicted survivors: {preds.sum()}/418 ({preds.mean():.2%})")

print("\n" + "=" * 60)
print("DONE! Upload outputs/submission_v2.csv to Kaggle")
print("=" * 60)
