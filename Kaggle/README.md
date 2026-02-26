# Kaggle Projects

Applied ML on real Kaggle datasets, covering the full workflow: EDA, preprocessing, model building, evaluation, and interpretation.

| Notebook | Problem Type | Methods | Best Result |
|----------|-------------|---------|-------------|
| [Titanic Survival](./Getting_Started_with_Titanic.ipynb) | Binary Classification | EDA, Random Forest | Survival prediction from passenger features |
| [Telco Customer Churn](./Telco_churn.ipynb) | Binary Classification | 7 models, Pipeline, Feature Importance | Logistic Regression: F1=0.626, ROC-AUC=0.861 |

---

## Titanic — Study of a Shipwreck

**File:** [`Getting_Started_with_Titanic.ipynb`](./Getting_Started_with_Titanic.ipynb)

**Problem:** Predict passenger survival from demographic and ticket features.

**Approach:**
- Exploratory analysis of survival rates by gender and class
- Feature encoding (`Pclass`, `Sex`, `SibSp`, `Parch`)
- Random Forest classifier (100 trees, max_depth=5)

**Key finding:** Women survived at 74% vs. 19% for men — gender is by far the strongest signal.

---

## Telco Customer Churn Prediction

**File:** [`Telco_churn.ipynb`](./Telco_churn.ipynb)

**Problem:** Predict which telecom customers are likely to churn (leave the company). Retaining existing customers is cheaper than acquiring new ones, so early identification of at-risk customers is high business value.

**Dataset:** 7,043 customers × 20 features (contract type, payment method, tenure, monthly charges, service subscriptions, etc.)

### EDA highlights

- Churn rate is ~26% (class imbalance addressed by tracking precision/recall alongside accuracy)
- **Tenure** is the strongest loyalty signal: long-tenure customers rarely churn
- **Month-to-month contracts** and **electronic check payments** correlate strongly with churn
- **Fiber optic service** customers churn at higher rates than DSL users
- Customers without tech support or device protection churn more

### Preprocessing

- Type coercion (`TotalCharges` stored as string → numeric; 11 nulls dropped)
- One-hot encoding of all categorical features
- `StandardScaler` applied inside a `sklearn.Pipeline` to prevent data leakage

### Model Comparison

Seven models trained and evaluated on the same 80/20 split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Logistic Regression** | **0.810** | **0.738** | 0.543 | **0.626** | **0.861** |
| Gradient Boosting | 0.803 | 0.726 | 0.523 | 0.608 | 0.858 |
| XGBoost | 0.802 | 0.727 | 0.518 | 0.605 | 0.859 |
| SVM | 0.799 | 0.732 | 0.491 | 0.588 | 0.802 |
| Random Forest | 0.796 | 0.730 | 0.479 | 0.579 | 0.830 |
| Decision Tree | 0.782 | 0.678 | 0.482 | 0.563 | 0.834 |
| K-Nearest Neighbors | 0.747 | 0.579 | 0.489 | 0.530 | 0.768 |

**Winner: Logistic Regression** — best F1 and ROC-AUC, and interpretable coefficients.

### Feature Importance

Top drivers of churn (from Logistic Regression coefficients):
1. **Tenure** — longer tenure strongly reduces churn probability
2. **Monthly charges** — higher charges increase churn risk
3. **Fiber optic internet** — associated with elevated churn vs. DSL

XGBoost Gini importance confirms tenure and monthly charges as the top two predictors.

### Business Recommendation

Customers most at risk: short tenure + high monthly charges + month-to-month contract + fiber optic service. Targeted interventions: discounted long-term contracts or complimentary tech support for this segment.
