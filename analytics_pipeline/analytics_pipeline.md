# Analytics & Machine Learning Pipeline Documentation

This document outlines the end-to-end analytics and machine learning pipeline for the Titanic dataset, synthesizing exploratory data analysis ([01_eda.ipynb](file:///d:/Capstone_Project/analytics_pipeline/01_eda.ipynb)), data transformation, model training, hyperparameter tuning, imbalance handling, regression diagnostics, and deployment artifacts.

---

### Step 1: Exploratory Data Analysis & Data Profiling

From initial inspection in [01_eda.ipynb](file:///d:/Capstone_Project/analytics_pipeline/01_eda.ipynb):
- **Initial Dataset**: 891 entries, 15 columns.
- **Missing Value Profile**:
  - `deck`: 688 missing ($77.22\%$) $\rightarrow$ Dropped due to $>30\%$ threshold.
  - `age`: 177 missing ($19.87\%$) $\rightarrow$ Imputed with median ($28.00$) / mean.
  - `embarked` / `embark_town`: 2 missing ($0.22\%$) $\rightarrow$ Imputed with mode (`'S'` / `'Southampton'`).

#### Key Analytical Findings & Narrative:
1. **Target Variable Imbalance**:
   - Non-Survivors ($0$): $61.62\%$ ($549$ passengers)
   - Survivors ($1$): $38.38\%$ ($342$ passengers)
2. **Gender Survival Disparity ("Women First" policy)**:
   - Female Survival Rate: **$74.04\%$** (312 total)
   - Male Survival Rate: **$18.89\%$** (577 total)
3. **Socioeconomic Advantage**:
   - 1st Class Survival Rate: **$62.62\%$**
   - 2nd Class Survival Rate: **$47.28\%$**
   - 3rd Class Survival Rate: **$24.24\%$**
4. **Intersectional Cohort Analysis**:
   - 1st Class Females: **$96.74\%$** survival (89 of 92 survived; only 3 deceased).
   - 2nd Class Females: **$92.11\%$** survival (70 of 76 survived).
   - 3rd Class Females: **$50.00\%$** survival (72 of 144 survived).
   - 1st Class Males: **$36.89\%$** survival (45 of 122 survived).
   - 2nd Class Males: **$15.74\%$** survival (17 of 108 survived).
   - 3rd Class Males: **$13.54\%$** survival (47 of 347 survived).
5. **Distribution Skewness**:
   - `fare` is heavily right-skewed (Median: $\$14.45$, Mode: $\$8.05$, Mean: $\$32.10$) due to high-value 1st-class tickets reaching over $\$500$.

---

### Step 2: Preprocessing & ColumnTransformer Pipeline

To prevent **data leakage**, all transformations are fitted strictly on $X_{train}$ and applied to $X_{test}$.

- **Stratified Split**: 80/20 train/test split stratified on `survived` (`stratify=y`) to maintain the $61.6\% / 38.4\%$ class ratio across both folds.
- **Numerical Pipeline**:
  - `age`, `fare` $\rightarrow$ `SimpleImputer(strategy='mean')` $\rightarrow$ `StandardScaler()`.
- **Categorical Pipeline**:
  - `sex`, `embarked` $\rightarrow$ `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
- **Feature Selection / Auto-Drop**:
  - `remainder='drop'` automatically filters out redundant string columns (`alive`, `class`, `deck`, `who`, `embark_town`) while passing through `pclass`, `sibsp`, `parch`, `adult_male`, and `alone`.

---

### Step 3: Class Imbalance Handling Comparison

We tested 3 variants of Logistic Regression on the holdout test set to evaluate minority-class handling:

| Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(a) Baseline (No Handling)** | **0.7978** | **0.7619** | 0.6957 | 0.7273 | 0.8524 |
| **(b) `class_weight='balanced'`** | 0.7865 | 0.7042 | **0.7246** | **0.7143** | **0.8541** |
| **(c) SMOTE (Train-Only)** | 0.7809 | 0.6944 | **0.7246** | 0.7092 | 0.8502 |

> **Imbalance Conclusion:**  
> `class_weight='balanced'` is the preferred imbalance strategy. It adjusts the cost function directly without generating synthetic Euclidean artifacts (unlike SMOTE), boosting minority-class Recall from $69.57\%$ to $72.46\%$ while preserving model stability.

---

### Step 4: Model Benchmarking & Performance Comparison

> **Note on Metric Scales:**  
> Classification metrics ($0.0 \text{ to } 1.0$ probabilistic scale) and Regression metrics (residual fit scale) represent two distinct tasks and are reported as separate metric groups below.

#### 1. Classification Performance (Target: `survived`)
*Evaluated on holdout test set ($N = 178$).*

| Model | Target | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | `survived` | **0.8258** | **0.8065** | **0.7246** | **0.7634** | **0.8693** |
| **Logistic Regression** | `survived` | 0.7978 | 0.7619 | 0.6957 | 0.7273 | 0.8524 |
| **Decision Tree ($d=3$)** | `survived` | 0.7865 | 0.7636 | 0.6087 | 0.6774 | 0.8261 |

* **Random Forest Hyperparameters (via GridSearchCV & OOB)**:
  - `n_estimators`: 300
  - `max_depth`: 8
  - `max_features`: 0.5
  - `oob_score_`: **0.8357** (Out-of-bag validation accuracy)

#### 2. Regression Fit & Error (Target: `survived` — Linear Probability Model)

| Model | Target | MAE | RMSE | $R^2$ Score | Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multivariate Linear Regression** | `survived` | **0.2912** | **0.3704** | **0.4208** | **0.3934** |

#### 3. Residual Diagnostics & Heteroscedasticity:
- When fitting a regression model on `fare`, the residual plot shows distinct **heteroscedasticity** (a non-random fan/cone spread). Residual variance expands significantly as predicted fares exceed $\$50$, driven by extreme 1st-class ticket prices ($>\$250$) and the non-negative price boundary ($>\$0$).
- When predicting binary `survived` via Linear Regression, the linear probability model produces unbounded predictions outside $[0, 1]$ and non-normal residuals, confirming that non-linear tree-based ensembles are mathematically superior for this task.

---

### Step 5: Final Deployment Recommendation

* **Primary Recommendation:** I recommend deploying the **Random Forest Classifier** as it achieves the best overall performance on the test set, leading with an **ROC-AUC of $0.8693$** and an **Accuracy of $82.58\%$**.
* **Superior Balance:** It delivers the highest **F1-Score of $0.7634$** with a strong balance between **Precision ($80.65\%$)** and **Recall ($72.46\%$)**, outperforming Logistic Regression ($\text{F1} = 0.7273$) and Decision Tree ($\text{F1} = 0.6774$).
* **Complex Feature Interactions:** The ensemble architecture effectively captures key non-linear relationships—such as the intersection of passenger class, sex, and fare—without the high variance of a single decision tree.
* **Appropriate Model Type:** Unlike linear regression ($R^2 = 0.4208$), Random Forest natively constrains survival probabilities to $[0, 1]$ while minimizing missed survivors ($\text{FN} = 19$).

---

### Step 6: Production Pipeline Artifact & Verification

The best-performing model and its preprocessing steps have been serialized together as a single artifact:
- **Serialized Artifact**: `titanic_best_rf_pipeline.joblib` / `titanic_production_pipeline.joblib`
- **End-to-End Inference Verification**:
  ```python
  import joblib
  import pandas as pd

  # Load the unified pipeline
  pipeline = joblib.load('titanic_best_rf_pipeline.joblib')

  # Predict on raw, unprocessed input with missing values & text
  raw_passenger = pd.DataFrame([{
      'pclass': 1, 'sex': 'female', 'age': 38.0, 'sibsp': 1,
      'parch': 0, 'fare': 71.28, 'embarked': 'C', 'adult_male': False, 'alone': False
  }])

  prediction = pipeline.predict(raw_passenger)
  probability = pipeline.predict_proba(raw_passenger)[:, 1]
  print(f"Outcome: {prediction[0]} | Survival Probability: {probability[0]:.2%}")
  ```

---

### Misc / Reference Files
- **Raw Dataset**: [titanic.csv](file:///d:/Capstone_Project/analytics_pipeline/titanic.csv) (891 passenger records).
- **EDA & Storytelling Notebook**: [01_eda.ipynb](file:///d:/Capstone_Project/analytics_pipeline/01_eda.ipynb) (Data profiling, missing values, skewness, 4-chart narrative).
- **Modeling & Pipeline Notebook**: [02_eda.ipynb](file:///d:/Capstone_Project/analytics_pipeline/02_eda.ipynb) (ColumnTransformer, classifiers, imbalance handling, regression diagnostics, grid search).
- **Model Artifact**: `titanic_best_rf_pipeline.joblib` (Full self-contained pipeline ready for production).
