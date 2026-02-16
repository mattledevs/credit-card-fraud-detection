# Credit Card Fraud Detection

## Problem Statement

This project builds machine learning models to detect fraudulent credit card transactions from a highly imbalanced dataset containing **284,807 transactions** with only **492 frauds (0.17%)**.

The challenge is to accurately identify the rare fraudulent transactions while minimizing false positives (blocking legitimate transactions) and false negatives (missing fraudulent ones).

## Dataset

- **Total Transactions:** 284,807
- **Fraudulent Cases:** 492 (0.17%)
- **Legitimate Cases:** 284,315 (99.83%)
- **Features:** 30 (PCA-transformed features V1-V28, plus Time and Amount)
- **Class Imbalance Ratio:** ~578:1

## Modeling Approach

### Imbalance Handling Techniques

1. **Class Weights:** Penalizes misclassification of minority class during training
2. **SMOTE:** Synthetically generates minority class samples to balance the training set
3. **Scale-Pos-Weight (XGBoost):** Weights positive class based on class distribution
4. **Stratified Train-Test Split:** Maintains class proportions in train/test sets
5. **Appropriate Evaluation Metrics:** Uses Precision, Recall, F1-Score, and ROC-AUC instead of Accuracy

### Models Trained

1. **Logistic Regression with Class Weights**
2. **Random Forest with Class Weights**
3. **XGBoost with Scale-Pos-Weight**
4. **Logistic Regression + SMOTE**

## Results

### Model Performance (ROC-AUC Score)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9978 | 0.9268 | 0.9701 | **0.9481** | **0.9681** |
| **LR + SMOTE** | **0.9769** | **0.9653** | **0.9787** | **0.9719** | **0.9672** |
| XGBoost | 0.9983 | 0.8943 | 0.9888 | 0.9381 | **0.9660** |
| Random Forest | 0.9999 | 0.9632 | 0.7230 | 0.8309 | 0.9382 |

**Best Performing Model:** Logistic Regression with class weights (ROC-AUC: **0.9681**)

### Key Performance Insights

- **Sensitivity (Recall):** 97.01% - Catches 97% of fraudulent transactions
- **Specificity:** 97.88% - Correctly identifies 97% of legitimate transactions
- **Precision:** 92.68% - Of predicted frauds, 92.68% are actual fraud
- All models achieved ROC-AUC > 0.96, indicating excellent discrimination ability

### Feature Importance

**Top Important Features (Random Forest):**
1. V14 (18.9%)
2. V10 (12.1%)
3. V4 (11.1%)
4. V12 (9.98%)
5. V17 (8.7%)

**Top Important Features (XGBoost):**
1. V14 (48.0%)
2. V4 (6.6%)
3. V12 (4.1%)
4. V10 (3.5%)
5. V20 (2.9%)

## Recommendations

1. **Model Deployment:** Use Logistic Regression or LR+SMOTE for production due to excellent F1-score balance
2. **Threshold Tuning:** Adjust fraud detection threshold based on business costs of false positives vs. false negatives
3. **Precision Priority:** If blocking legitimate transactions is costly, optimize for higher precision
4. **Recall Priority:** If missing fraud is costly, optimize for higher recall
5. **Continuous Monitoring:** Implement model monitoring in production to detect performance degradation
6. **Model Retraining:** Periodically retrain with new data to adapt to evolving fraud patterns

## Files

- `creditcard.csv` - Original dataset (Kaggle)
- `data_exploration.ipynb` - Exploratory data analysis with visualizations
- `fraud_detection_model.ipynb` - Model development, training, and evaluation
- `README.md` - This file

## Libraries Used

- pandas, numpy - Data manipulation
- scikit-learn - ML algorithms and utilities
- xgboost - Gradient boosting
- imbalanced-learn - SMOTE for handling class imbalance
- matplotlib, seaborn - Visualization

---

**Project Status:** Complete | **Date:** February 2026
