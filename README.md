# 📘 Supervised Machine Learning with Python

## 📌 Project Overview

Notebook ini berisi eksplorasi dan implementasi **model supervised machine learning** menggunakan Python, dengan fokus pada:
- **Regresi Linier (Linear Regression)**
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Naive Bayes**

Dataset yang digunakan adalah:
- **USA Housing Dataset**: digunakan untuk prediksi harga rumah dengan pendekatan regresi.
- **Titanic Dataset**: digunakan untuk klasifikasi kelangsungan hidup penumpang.
- Dataset sintetik untuk klasifikasi biner menggunakan `make_classification`.

## 🔧 Tools & Libraries

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (sklearn)

## 🧠 Machine Learning Models

Notebook ini mencakup implementasi dari berbagai model supervised learning:

| Model               | Tipe        | Dataset                  | Evaluasi                         |
|--------------------|-------------|--------------------------|----------------------------------|
| Linear Regression   | Regression  | USA Housing Dataset      | MAE, MSE, RMSE, R² Score         |
| K-Nearest Neighbors | Classification | Titanic Dataset        | Confusion Matrix, Accuracy Score|
| SVM (SVC)           | Classification | Titanic Dataset        | Confusion Matrix, Accuracy Score|
| Decision Tree       | Classification | Titanic Dataset        | Feature Importance, Accuracy    |
| Random Forest       | Classification | Titanic Dataset        | Feature Importance, Accuracy    |
| GaussianNB          | Classification | Titanic Dataset        | Confusion Matrix, Accuracy Score|

## 📈 Evaluation Metrics

Untuk regresi:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Untuk klasifikasi:
- Confusion Matrix
- Accuracy
- Classification Report

## 💡 Insights

- Model **Random Forest** menunjukkan performa yang baik pada tugas klasifikasi dengan akurasi yang stabil dan fitur importance yang dapat diinterpretasikan.
- Model **Linear Regression** cocok untuk prediksi harga rumah dan memberikan interpretasi linier sederhana terhadap fitur.
- Perbandingan antara berbagai model klasifikasi memberikan pemahaman tentang bias-variance trade-off serta pentingnya preprocessing data.
- Visualisasi seperti heatmap korelasi, distribusi target, dan feature importance sangat membantu dalam memahami pola data.
