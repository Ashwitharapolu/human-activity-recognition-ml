# 🏃 Human Activity Recognition using Machine Learning

> Binary classification of **Walk vs Run** from wearable sensor data using 5 ML models — achieving **99.24% accuracy** with MLP Neural Network on 88,588 samples.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![MLP](https://img.shields.io/badge/MLP-99.24%25_Accuracy-brightgreen)

---

## 🌍 Live Demo
👉 [Click here to try the app](https://human-activity-recognition-ml-z9nsdr5pxtn2alnqww977n.streamlit.app)

---

## 🎯 Project Overview

This project builds a **Human Activity Recognition (HAR)** system that classifies whether a person is **Walking or Running** using raw accelerometer and gyroscope data from a wrist-worn sensor.

| Item | Details |
|------|---------|
| **Dataset** | 88,588 rows × 11 columns |
| **Features** | 9 features (6 sensor + 2 magnitude + wrist) |
| **Target** | Binary — Walk (0) vs Run (1) |
| **Best Model** | MLP Neural Network — **99.24% accuracy** |

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
| `acc_magnitude` | Magnitude of acceleration vector |
| `gyro_magnitude` | Magnitude of gyroscope vector |
| `wrist` | Which wrist (0 = left, 1 = right) |

---

## 🏗️ Project Pipeline

**Raw Sensor Data (88,588 rows)**
→ Exploratory Data Analysis
→ Feature Engineering (acc_magnitude + gyro_magnitude)
→ Feature Selection (t-test + Random Forest Importance)
→ Train-Test Split (80/20 Stratified)
→ Feature Scaling (StandardScaler)
→ Model Training (5 Models)
→ Hyperparameter Tuning (GridSearchCV)
→ Cross Validation (2-Fold)
→ ROC-AUC Evaluation
→ Best Model Saved (.pkl) + Deployed on Streamlit ✅
---

## 🤖 Model Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| Logistic Regression | 95.33% | Linear baseline model |
| Decision Tree | 98.25% | No scaling needed |
| Random Forest | 98.99% | 100 trees ensemble |
| KNN (k=5) | 99.16% | Distance based |
| **MLP Neural Network** | **99.24%** | **Best model — deployed** |
| Random Forest (Tuned) | 99.01% | GridSearchCV tuned |
| KNN (Tuned) | 99.16% | GridSearchCV tuned |

---

## 🔬 Key Techniques Used

- **Feature Engineering** — created acc_magnitude and gyro_magnitude from raw sensor axes
- **Statistical Feature Selection** — t-test to identify most discriminative features
- **Random Forest Feature Importance** — ranked all features by importance score
- **Stratified Train-Test Split** — 80/20 split maintaining class balance
- **Standard Scaling** — applied to models that need it (LR, KNN, MLP)
- **GridSearchCV** — hyperparameter tuning for Random Forest and KNN
- **Cross Validation** — 2-fold CV on best model (Mean: 98.90%, Std: 0.04%)
- **ROC-AUC Analysis** — model comparison beyond accuracy
- **Model Serialization** — saved model and scaler as .pkl using joblib

---

## 📈 Cross Validation Results

| Metric | Value |
|--------|-------|
| CV Fold 1 | 98.86% |
| CV Fold 2 | 98.94% |
| **Mean CV Accuracy** | **98.90%** |
| Standard Deviation | 0.04% |

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
| **Streamlit** | Interactive web app |

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `human_activity_recognition.ipynb` | Complete analysis notebook |
| `app.py` | Streamlit web app |
| `walkrun.csv` | Dataset |
| `walkrun_best_model.pkl` | Saved MLP model |
| `scaler.pkl` | Saved StandardScaler |
| `cm_best_model.png` | Confusion matrix plot |
| `requirements.txt` | Dependencies |
| `README.md` | This file |
---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Ashwitharapolu/human-activity-recognition-ml.git
cd human-activity-recognition-ml
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open browser at**

---

## 🔮 Use the Saved Model

```python
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load('walkrun_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sample input
# [wrist, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, acc_magnitude, gyro_magnitude]
raw_input = np.array([[0, 0.265, -0.781, -0.007, -0.059, 0.032, -2.929, 0.831, 2.931]])

# Scale and predict
scaled_input = scaler.transform(raw_input)
prediction = model.predict(scaled_input)

print("Walking" if prediction[0] == 0 else "Running")
```

---

## 🔍 Key Findings

- **Running generates significantly higher sensor variance** than walking
- **Gyroscope Z-axis** showed highest discriminative power
- **MLP Neural Network** outperformed all traditional ML models
- All models achieved above **95% accuracy**
- **acc_magnitude and gyro_magnitude** were the most important engineered features

---

## ⚠️ Limitations

- Only two activity classes — Walk and Run
- Single dataset source — may not generalise to all users
- Wrist-only sensor — multi-sensor setup would improve accuracy

---

## 🔮 Future Scope

- Expand to multi-class — cycling, stairs, sitting, standing
- Real-time prediction pipeline on smartphone
- Deep learning — 1D-CNN and LSTM for temporal patterns
- Multi-user data collection for generalisation

---

## 📊 Resume Highlights

- Trained and compared 5 ML classifiers on 88,588 wearable sensor samples achieving 99.24% accuracy with MLP Neural Network
- Applied feature engineering creating acceleration and gyroscope magnitude features improving model discriminability
- Used GridSearchCV hyperparameter tuning and 2-fold cross validation achieving 98.90% mean CV accuracy with 0.04% standard deviation
- Deployed interactive Streamlit app with real-time Walk vs Run predictions from sensor input values

---

## 👤 Author

**Ashwitha Rapolu**
- GitHub: [@Ashwitharapolu](https://github.com/Ashwitharapolu)

---

*Built with using Python + Scikit-learn + Streamlit*
