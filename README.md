# Credit Card Default Prediction using Machine Learning

---

## a. Problem Statement

The objective of this project is to build and compare multiple **machine learning classification models** to predict whether a credit card customer will **default on their payment in the next month**.  
Accurate default prediction is crucial for financial institutions to assess credit risk, minimize financial losses, and support informed decision-making.

In this project, six different classification models are implemented, evaluated using standard performance metrics, and deployed through an interactive **Streamlit web application**.

---

## b. Dataset Description

- **Dataset Name:** Default of Credit Card Clients  
- **Source:** UCI Machine Learning Repository  
- **Problem Type:** Binary Classification  

### Target Variable
- **`default payment next month`**
  - `0` → No default  
  - `1` → Default  

### Dataset Characteristics
- **Number of Instances:** 30,000  
- **Number of Features:** 23  
- **Feature Types:** Numerical and categorical (demographic details, billing amounts, payment history)

The dataset is **imbalanced**, with fewer default cases compared to non-default cases. Therefore, metrics such as **AUC** and **Matthews Correlation Coefficient (MCC)** are particularly important for model evaluation.

---

## c. Models Used and Evaluation Metrics

The following six machine learning classification models were implemented on the same dataset and evaluated using the required performance metrics.

### Comparison Table of Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.678167 | 0.716561 | 0.367660 | 0.632253 | 0.464949 | 0.276528 |
| Decision Tree | 0.731333 | 0.612364 | 0.394052 | 0.399397 | 0.396707 | 0.223912 |
| kNN | 0.810333 | 0.728508 | 0.625166 | 0.355690 | 0.453410 | 0.369317 |
| Naive Bayes | 0.290333 | 0.724002 | 0.233981 | 0.971364 | 0.377121 | 0.103419 |
| Random Forest (Ensemble) | 0.793167 | 0.773047 | 0.532186 | 0.535795 | 0.533984 | 0.401078 |
| XGBoost (Ensemble) | 0.817333 | 0.770592 | 0.656716 | 0.364732 | 0.468992 | 0.392711 |

---

### Model-wise Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Demonstrated good recall and moderate AUC, indicating effective identification of defaulters, but precision and MCC were relatively low due to class imbalance. |
| Decision Tree | Provided interpretable results but showed lower AUC and MCC, suggesting limited generalization and weaker performance on imbalanced data. |
| kNN | Achieved high accuracy but low recall, indicating a strong bias toward the majority (non-default) class and missed many defaulters. |
| Naive Bayes | Achieved very high recall, successfully identifying most defaulters, but suffered from extremely low precision and accuracy due to over-prediction of the default class. |
| Random Forest (Ensemble) | Delivered a strong balance across accuracy, AUC, F1 score, and MCC, making it one of the most reliable models for this dataset. |
| XGBoost (Ensemble) | Achieved the highest accuracy along with strong AUC and MCC values, indicating robust performance and effective handling of complex feature interactions. |

---

## Summary

The experimental results show that **ensemble models (Random Forest and XGBoost)** outperform individual classifiers on this dataset.  
XGBoost achieved the highest overall accuracy, while Random Forest provided consistently strong performance across all evaluation metrics.  
These findings highlight the effectiveness of ensemble learning techniques for handling **imbalanced financial datasets** such as credit card default prediction.

---