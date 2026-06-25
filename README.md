# 🏃 Human Activity Recognition using Machine Learning

> Binary classification of **Walk vs Run** from wearable sensor data using 5 ML models — achieving **99.24% accuracy** with MLP Neural Network on 88,588 samples.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)

---

## 🎯 Project Overview

This project builds a **Human Activity Recognition (HAR)** system that classifies whether a person is **Walking or Running** using raw accelerometer and gyroscope data from a wrist-worn sensor.

| Item | Details |
|------|---------|
| **Dataset** | 88,588 rows × 11 columns |
| **Features** | 6 sensor features (3-axis accelerometer + 3-axis gyroscope) |
| **Target** | Binary — Walk (0) vs Run (1) |
| **Best Model** | MLP Neural Network — **99.24% accuracy** |
| **Dataset Source** | Wearable sensor data (walkrun.csv) |

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `acceleration_x` | Wrist acceleration — X axis |
| `acceleration_y` | Wrist acceleration — Y axis |
| `acceleration_z` | Wrist acceleration — Z axis |
| `gyro_x` | Gyroscope reading — X axis |
| `gyro_y` | Gyroscope reading — Y axis |
| `gyro_z` | Gyroscope reading — Z axis |
| `wrist` | Which wrist (0 = left, 1 = right) |
| `activity` | Target — 0 = Walk, 1 = Run |

---

## 🏗️ Project Pipeline

```
Raw Sensor Data (88,588 rows)
        ↓
Exploratory Data Analysis
        ↓
Feature Selection (t-test + Random Forest Importance)
        ↓
Train-Test Split (80/20 Stratified)
        ↓
Feature Scaling (StandardScaler for LR, KNN, MLP)
        ↓
Model Training (5 Models)
        ↓
Hyperparameter Tuning (GridSearchCV)
        ↓
Cross Validation (2-Fold)
        ↓
ROC-AUC Evaluation
        ↓
Best Model Saved (.pkl)
```

---

## 🤖 Model Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| Logistic Regression | 95.33% | Linear baseline model |
| Decision Tree | 98.25% | No scaling needed |
| Random Forest | 98.99% | 100 trees ensemble |
| KNN (k=5) | 99.16% | Distance based |
| **MLP Neural Network** | **99.24%** | **Best model** |
| Random Forest (Tuned) | 99.01% | GridSearchCV tuned |
| KNN (Tuned) | 99.16% | GridSearchCV tuned |

---

## 🔬 Key Techniques Used

- **Exploratory Data Analysis** — distribution plots, correlation heatmap, class balance check
- **Statistical Feature Selection** — t-test to identify most discriminative features between Walk and Run
- **Random Forest Feature Importance** — ranked all 6 sensor features by importance score
- **Stratified Train-Test Split** — 80/20 split maintaining class balance
- **Standard Scaling** — applied only to models that need it (LR, KNN, MLP)
- **GridSearchCV** — hyperparameter tuning for Random Forest and KNN
- **Cross Validation** — 2-fold CV on best model (Mean: 98.90%, Std: 0.04%)
- **ROC-AUC Analysis** — model comparison beyond just accuracy
- **Confusion Matrix** — error analysis for best model
- **Model Serialization** — saved best model as `.pkl` using joblib

---

## 📈 Cross Validation Results

```
Cross Validation Scores : [0.9886, 0.9894]
Mean CV Accuracy        : 98.90%
Standard Deviation      : 0.04%
```

Very low standard deviation confirms the model is **stable and not overfitting.**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9** | Core language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computations |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **Matplotlib** | Visualizations |
| **Seaborn** | Statistical plots |
| **SciPy** | t-test for feature selection |
| **Joblib** | Model serialization |

---

## 📁 Project Structure

```
human-activity-recognition/
├── human_activity_recognition.ipynb  ← Complete analysis notebook
├── walkrun.csv                        ← Dataset
├── walkrun_best_model.pkl             ← Saved best model
├── cm_best_model.png                  ← Confusion matrix plot
└── README.md                          ← This file
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Ashwitharapolu/human-activity-recognition.git
cd human-activity-recognition
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib
```

**3. Open the notebook**
```bash
jupyter notebook human_activity_recognition.ipynb
```

**4. Run all cells** — top to bottom

---

## 🔮 Use the Saved Model

```python
import joblib
import numpy as np

# Load saved model
model = joblib.load('walkrun_best_model.pkl')

# Predict on new sensor reading
# [acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, wrist]
sample = np.array([[0.265, -0.781, -0.007, -0.059, 0.032, -2.929, 0]])
prediction = model.predict(sample)

print("Walking" if prediction[0] == 0 else "Running")
```

---

## 🔍 Key Findings

- **Running generates significantly higher sensor variance** than walking across all 6 axes
- **Gyroscope Z-axis** showed the highest discriminative power between activities
- **MLP Neural Network** outperformed all traditional ML models
- All models achieved above **95% accuracy** — confirming sensor data is highly informative
- **Logistic Regression at 95.33%** proves even a simple linear model captures the pattern well

---

## ⚠️ Limitations

- Only two activity classes — Walk and Run
- Single participant (viktor) — may not generalise to other users
- Pre-collected static data — not tested in real-time
- Wrist-only sensor — multi-sensor setup would improve accuracy

---

## 🔮 Future Scope

- Expand to multi-class — cycling, stairs, sitting, standing
- Real-time prediction pipeline on Raspberry Pi or smartphone
- Deep learning — 1D-CNN and LSTM for temporal pattern recognition
- Multi-user data collection for generalisation
- TensorFlow Lite deployment for Android/iOS

---

## 📊 Resume Highlights

- Trained and compared 5 ML classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, MLP) on 88,588 wearable sensor samples for Walk vs Run classification
- Achieved 99.24% test accuracy with MLP Neural Network using feature selection via t-test and Random Forest importance scores
- Applied GridSearchCV hyperparameter tuning and 2-fold stratified cross validation achieving 98.90% mean CV accuracy with 0.04% standard deviation

---

## 👤 Author

**Ashwitha Rapolu**
- GitHub: [@Ashwitharapolu](https://github.com/Ashwitharapolu)

---

*Built with ❤️ using Python + Scikit-learn*
