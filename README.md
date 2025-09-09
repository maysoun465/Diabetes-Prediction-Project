# Diabetes Prediction Project

## Project Overview
This project is focused on predicting diabetes in patients using various machine learning models. Early detection of diabetes is crucial for improving patient outcomes, and this project aims to build a robust predictive system using clinical measurements.

The project follows a structured workflow:

1. **Exploratory Data Analysis (EDA)**  
   - Understand dataset structure, distribution, and correlations.
   - Visualize key features like Glucose, BMI, Age, and their relationship with diabetes outcome.

2. **Data Preprocessing & Feature Engineering**  
   - Standardize features.
   - Create interaction features like `Glucose_BMI`, `Glucose_Age`, `BMI_DPF`.
   - Split data into training and testing sets.

3. **Model Training & Hyperparameter Tuning**  
   - Models used:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
     - XGBoost
     - CatBoost
     - Voting Classifier
     - Stacking Classifier
   - Hyperparameter tuning applied (GridSearchCV) for better performance.
   - Emphasis on **Recall** due to medical importance (minimizing false negatives).

4. **Model Evaluation**  
   - Metrics considered: Accuracy, Precision, Recall, F1-score, AUC
   - Confusion matrices and ROC curves used to assess performance.

---

## Dataset
- Source: `diabetes.csv`  
- Features: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`  
- Target: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

---

## Model Evaluation Table

| Model                 | Accuracy | Precision | Recall  | F1-score | AUC   |
| --------------------- | -------- | --------- | ------- | -------- | ----- |
| Stacking              | 0.779    | 0.706     | 0.655   | 0.679    | 0.836 |
| Random Forest         | 0.766    | 0.656     | 0.727   | 0.690    | 0.839 |
| Logistic Regression   | 0.766    | 0.673     | 0.673   | 0.673    | 0.814 |
| XGBoost               | 0.747    | 0.618     | 0.764   | 0.683    | 0.790 |
| CatBoost              | 0.747    | 0.629     | 0.709   | 0.667    | 0.812 |
| Voting                | 0.753    | 0.639     | 0.709   | 0.672    | 0.837 |
| **SVM Tuned (SVMTT)** | 0.714    | 0.573     | 0.782   | 0.662    | 0.810 |

### Observations:
- **SVM Tuned (SVMTT)** achieved the highest Recall (0.782), critical in medical applications.
- Models like **XGBoost** and **Random Forest** also show high Recall despite slightly lower Accuracy.
- **Stacking** has the highest Accuracy (0.779) but lower Recall (0.655).
- Final model choice depends on balancing **Accuracy and Recall**.
