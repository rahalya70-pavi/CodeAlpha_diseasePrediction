# CodeAlpha_DiseasePrediction

## Breast Cancer Prediction using Machine Learning
CodeAlpha Internship - Machine Learning Task 4: Disease Prediction from Medical Data

## Objective
Predict whether a tumor is malignant or benign using patient medical data.

## Dataset
Wisconsin Breast Cancer Dataset (built into scikit-learn)
- 569 samples, 30 features
- Target: malignant (0) or benign (1)

## Approach
- Data preprocessing and feature scaling (StandardScaler)
- Trained 3 classification models: Logistic Regression, Random Forest, SVM
- Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Feature importance analysis using Random Forest

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 98.2% | 98.6% | 98.6% | 98.6% | 0.995 |
| Random Forest | 95.6% | 95.9% | 97.2% | 96.6% | 0.993 |
| SVM | 98.2% | 98.6% | 98.6% | 98.6% | 0.995 |

## Visualizations
- Model performance comparison
- Confusion matrix
- ROC curve
- Feature importance chart

## Tech Stack
Python, scikit-learn, pandas, matplotlib, seaborn

## How to Run
