# Diabetes Prediction Project 

## 📌 Project Overview
This project aims to **predict diabetes in patients** using machine learning models.  
The goal is to explore the dataset, perform feature engineering, train multiple models, and evaluate their performance for **early diabetes detection**.

---

## 📂 Repository Structure
- `diabetes.csv` → The raw dataset containing patient medical data  
- `Diabetes.ipynb` → Jupyter Notebook containing all steps:
  - Exploratory Data Analysis (EDA)  
  - Data Visualization  
  - Data Preprocessing & Feature Engineering  
  - Train/Test Split & Scaling  
  - Model Training & Hyperparameter Tuning  
  - Model Evaluation  
  - Diabetes Prediction Engine
- `model_performance_comparison.png` → Heatmap of model performance metrics
- `README.md` → Project description and instructions  

---

## ⚙️ Prerequisites
- Python 3.x  
- Jupyter Notebook or Google Colab  

**Python Libraries:**
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- catboost  

---

## 🔑 Key Steps

### 1️⃣ Exploratory Data Analysis (EDA)
- Summary statistics  
- Missing values check  
- Correlation analysis  
- Outcome distribution visualization  
- Feature distribution and relationship plots (`Glucose`, `BMI`, `Age`)  

### 2️⃣ Data Preprocessing & Feature Engineering
- Dropped weak/noisy features: `SkinThickness`, `BloodPressure`  
- Created interaction features:
  - `Glucose_BMI = Glucose * BMI`  
  - `Glucose_squared = Glucose²`  
  - `BMI_squared = BMI²`  
  - `Glucose_Age = Glucose * Age`  
  - `BMI_DPF = BMI * DiabetesPedigreeFunction`  
- StandardScaler applied for feature scaling  
- Train/Test split: **80/20**  

### 3️⃣ Model Training & Hyperparameter Tuning
Models used:  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  
- CatBoost  
- Stacking Classifier
  
Hyperparameter tuning with **GridSearchCV** for:
- SVM  
- Logistic Regression  
- Random Forest
- LinearSVC

---

## 📊 Model Evaluation
| Model                   | Accuracy | Precision | Recall  | F1-score | AUC   |
| ----------------------- | -------- | --------- | ------- | -------- | ----- |
| **Stacking**            | 0.877    | 0.810     | 0.855   | 0.832    | 0.929 |
| **Random Forest Tuned** | 0.877    | 0.810     | 0.855   | 0.832    | 0.933 |
| **CatBoost**            | 0.877    | 0.821     | 0.836   | 0.829    | 0.935 |
| SVM                     | 0.870    | 0.830     | 0.800   | 0.815    | 0.887 |
| Random Forest           | 0.864    | 0.793     | 0.836   | 0.814    | 0.930 |
| XGBoost                 | 0.851    | 0.786     | 0.800   | 0.793    | 0.935 |
| **SVM Tuned**           | 0.818    | 0.680     | 0.927   | 0.785    | 0.912 |
| Logistic Regression     | 0.818    | 0.737     | 0.764   | 0.750    | 0.869 |
| Linear SVC              | 0.805    | 0.667     | 0.909   | 0.769    | 0.867 |
| Logistic Regression Tuned | 0.766  | 0.620     | 0.891   | 0.731    | 0.877 |

### Model Performance Metrics Comparison

![Model Performance Metrics Comparison](model_performance_comparison.png)

---

## 📈 Key Insights  
- Higher **Glucose** and **BMI** strongly correlate with diabetes.  
- Older age increases diabetes risk.  
- **SVM Tuned** achieved the highest **Recall (0.927)** → crucial for minimizing false negatives in medical datasets.  
- **XGBoost** and **Random Forest** maintain high Recall despite slightly lower Accuracy.  
- **Stacking Classifier** and **Random Forest Tuned** achieved the highest Accuracy (~0.877).  
- Final model selection depends on balancing **Accuracy** and **Recall** according to project goals. 

---

## 🚀 Diabetes Prediction Engine
- Predicts diabetes for new patients using the engineered features and trained model.  
- Features required:  
  `Pregnancies`, `Glucose`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`  

```python
prediction = predict_diabetes_final(new_patient_data)
