# Credit Card Default Prediction using Machine Learning

---

## Student Information

- **Name:** GANESAN V
- **BITS ID:** 2025AA05382
- **Course:** Machine Learning Assignment 2
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
| Logistic Regression | 0.678167 | 0.710561 | 0.367660 | 0.632253 | 0.464949 | 0.276528 |
| Decision Tree | 0.731333 | 0.612364 | 0.394052 | 0.399397 | 0.396707 | 0.223912 |
| kNN | 0.810333 | 0.728508 | 0.625166 | 0.355690 | 0.453410 | 0.369317 |
| Naive Bayes | 0.290333 | 0.724002 | 0.233981 | 0.971364 | 0.377121 | 0.103419 |
| Random Forest (Ensemble) | 0.793167 | 0.773047 | 0.532186 | 0.535795 | 0.533984 | 0.401078 |
| XGBoost (Ensemble) | 0.817333 | 0.770592 | 0.656716 | 0.364732 | 0.468992 | 0.392711 |


---

### Model-wise Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Demonstrated good recall (0.632) and moderate AUC (0.711), indicating effective identification of defaulters, but precision (0.368) and MCC (0.277) were relatively low due to class imbalance. |
| Decision Tree | Provided interpretable results with moderate accuracy (0.731) but showed the lowest AUC (0.612) and relatively low MCC (0.224), suggesting limited generalization and weaker performance on imbalanced data. |
| kNN | Achieved high accuracy (0.810) and good AUC (0.729) but low recall (0.356), indicating a strong bias toward the majority (non-default) class and missed many defaulters despite high precision (0.625). |
| Naive Bayes | Achieved the highest recall (0.971), successfully identifying most defaulters, but suffered from extremely low precision (0.234) and the lowest accuracy (0.290) due to over-prediction of the default class, resulting in the lowest MCC (0.103). |
| Random Forest (Ensemble) | Delivered the best overall balance with the highest AUC (0.773), MCC (0.401), and well-balanced precision (0.532) and recall (0.536), making it the most reliable and stable model for this dataset. |
| XGBoost (Ensemble) | Achieved the highest accuracy (0.817) and precision (0.657) with strong AUC (0.771) and MCC (0.393), indicating robust performance and effective handling of complex feature interactions, though recall (0.365) was moderate. |

---

## Summary

The experimental results show that **ensemble models (Random Forest and XGBoost)** significantly outperform individual classifiers on this imbalanced dataset.  

**Random Forest** emerged as the most reliable model overall, achieving the highest AUC (0.773) and MCC (0.401) with excellent balance between precision (0.532) and recall (0.536). This balanced performance makes it particularly suitable for credit default prediction where both false positives and false negatives have significant costs.

**XGBoost** achieved the highest accuracy (0.817) and precision (0.657), demonstrating exceptional ability to correctly identify non-defaulters and minimize false alarms. However, its lower recall (0.365) indicates it may miss more actual defaulters compared to Random Forest.

Among individual classifiers, **kNN** performed best with high accuracy (0.810) and good AUC (0.729), though its low recall (0.356) suggests bias toward the majority class. **Naive Bayes** achieved exceptional recall (0.971), capturing almost all defaulters, but at the cost of very low precision (0.234) and accuracy (0.290), making it impractical for production use.

These findings highlight the effectiveness of **ensemble learning techniques** for handling imbalanced financial datasets, with Random Forest offering the best overall trade-off between all evaluation metrics for credit card default prediction.


---

