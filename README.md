# Breast-Cancer-Prediction-Using-Machine-Learning
This project focuses on building machine learning models to predict whether a breast tumor is malignant (M) or benign (B) using clinical diagnostic features.  The Wisconsin Breast Cancer dataset was used, which contains measurements computed from digitized images of fine needle aspirates (FNA) of breast masses. 
Two supervised learning algorithms were implemented and compared:

Logistic Regression

Support Vector Machine (SVM)

📂 Dataset Information

Source: Kaggle – Breast Cancer Dataset

Total Samples: 569

Features: 30 numerical diagnostic features

Target Variable:

M → Malignant (Cancerous)

B → Benign (Non-cancerous)

Class distribution:

Benign: 357

Malignant: 212

⚙️ Workflow

The project follows a standard machine learning pipeline:

1️⃣ Data Loading

CSV file loaded using Pandas.

Dropped unnecessary identifier column (id).

2️⃣ Target Encoding

Converted diagnosis labels:

M → 1

B → 0

3️⃣ Train–Test Split

Dataset split into:

80% Training

20% Testing

Stratified sampling used to preserve class balance.

4️⃣ Feature Scaling

Standardization applied using StandardScaler.

Required because models like Logistic Regression and SVM are sensitive to feature scale.

🤖 Models Trained
🔵 Logistic Regression

Linear classification model.

Good baseline for medical prediction tasks.

Accuracy achieved:
👉 96.49%

🟣 Support Vector Machine (SVM)

Non-linear classifier with RBF kernel.

Capable of capturing complex decision boundaries.

Accuracy achieved:
👉 97.36%

📊 Model Comparison
Model	Accuracy
Logistic Regression	96.49%
Support Vector Machine	97.36%

✔️ SVM performed slightly better than Logistic Regression on the test dataset.
