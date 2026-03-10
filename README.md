**🩺 Breast Cancer Prediction Using Machine Learning**(https://breast-cancer-prediction-using-machine-learning-bnthfrjtasnde9.streamlit.app/)

This project implements a machine learning–based system to predict whether a breast tumor is malignant or benign using diagnostic medical data.

Using the well-known Wisconsin Breast Cancer dataset, multiple classification models were trained and evaluated to determine the most effective approach. The final system achieves ~97% accuracy, demonstrating the potential of machine learning for assisting early cancer detection.

**🎯 Objectives**

Analyze and preprocess clinical breast cancer data

Train and evaluate classification models

Compare performance between algorithms

Build a prediction system for new patient records

Demonstrate an end-to-end ML workflow suitable for real-world healthcare applications

⚙️ Project Workflow

The project follows a standard machine learning pipeline:

🔹 1. Data Loading & Cleaning

Loaded dataset using Pandas

Removed unnecessary ID column

Checked for missing values

🔹 2. Label Encoding

Converted categorical diagnosis values:

M → 1

B → 0

🔹 3. Train–Test Split

Dataset split into 80% training and 20% testing

Stratified sampling used to preserve class balance

🔹 4. Feature Scaling

Applied StandardScaler to normalize feature ranges

Essential for distance-based models like SVM

🤖 Models Implemented

Two supervised learning algorithms were trained:

✅ Logistic Regression

Linear classification baseline

Fast and interpretable

Accuracy: ~96.5%

✅ Support Vector Machine (SVM)

RBF kernel used for non-linear classification

Captures complex boundaries

Accuracy: ~97.3%

📊 Performance Comparison
Model	Accuracy
Logistic Regression	96.49%
Support Vector Machine	97.36%

🏆 SVM achieved the best performance on this dataset.
