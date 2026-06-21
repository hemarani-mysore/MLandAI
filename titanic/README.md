# 🚢 Titanic Survival Prediction

Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

End-to-end ML pipeline covering EDA → Feature Engineering → Ensemble Modeling → Submission.

---

## 📊 Results

| Model | CV Accuracy (5-fold) |
|---|---|
| Logistic Regression | 81.03% |
| Random Forest | 83.28% |
| XGBoost | 83.95% |
| **Voting Ensemble** | **84.17%** ✅ |

---

## 🗂️ Project Structure

```
titanic/
├── data/
│   ├── train.csv           # Labeled training data (891 rows)
│   ├── test.csv            # Unlabeled test data (418 rows)
│   └── gender_submission.csv  # Kaggle's sample submission
├── src/
│   └── titanic_pipeline.py # Full ML pipeline (EDA → Submission)
└── outputs/
    ├── submission.csv       # Upload this to Kaggle
    ├── eda_plots.png        # Survival pattern visualizations
    └── feature_importance.png # Feature importance from Random Forest
```

---

## 🔍 Key Findings (EDA)

- **Sex** is the strongest raw predictor: women survived at ~74%, men at ~19%
- **Passenger Class** matters: 1st class ~63% vs 3rd class ~24%
- **Title** (extracted from Name) is the #1 engineered feature
- **Cabin** was 77% missing — extracted deck letter and "known/unknown" flag

---

## ⚙️ Feature Engineering

| Feature | Source | Rationale |
|---|---|---|
| `Title` | `Name` | Mr/Mrs/Miss/Master encodes age, gender, and social status |
| `FamilySize` | `SibSp + Parch + 1` | Small families had better survival odds |
| `IsAlone` | `FamilySize == 1` | Traveling alone was a disadvantage |
| `CabinKnown` | `Cabin` not null | Having a cabin number signals wealth |
| `Deck` | First letter of `Cabin` | Deck position affected lifeboat access |
| `FareBin` | Quartile-binned `Fare` | Reduces sensitivity to fare outliers |
| `AgeBin` | Age bucketed by life stage | Children had priority on lifeboats |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Run the pipeline
python titanic/src/titanic_pipeline.py
```

The pipeline will:
1. Print EDA insights to the console
2. Save plots to `outputs/`
3. Generate `outputs/submission.csv` ready for Kaggle upload

---

## 🧠 ML Concepts Covered

- **EDA** — understanding data distributions and patterns before modeling
- **Feature Engineering** — extracting signal from raw fields (Name → Title)
- **Imputation** — filling missing values (Age → median, Embarked → mode)
- **Label Encoding** — converting categorical strings to integers
- **StandardScaler** — normalizing features for Logistic Regression
- **Data Leakage** — why we `fit_transform` train but only `transform` test
- **Cross-Validation** — StratifiedKFold for reliable generalization estimate
- **Ensemble (Voting)** — combining models to reduce individual variance
- **Feature Importance** — interpreting which inputs drive the model's decisions
