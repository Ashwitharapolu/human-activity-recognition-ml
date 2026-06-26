# Human Activity Recognition - Streamlit App
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("walkrun_best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #f8fafc;
    }
    section[data-testid="stSidebar"] {
        background-color: #eef2ff;
        border-right: 1px solid #c7d2fe;
    }
    .stButton > button {
        background-color: #6366f1;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #4f46e5;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## 🏃 Human Activity Recognition")
st.markdown("Predict whether a person is **Walking** or **Running** from wearable sensor data")
st.divider()

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🤖 Best Model", "MLP Neural Network")
with col2:
    st.metric("🎯 Accuracy", "99.24%")
with col3:
    st.metric("📊 Dataset Size", "88,588 rows")
with col4:
    st.metric("🔢 Features", "9 Features")

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Sensor Input")
    st.markdown("*Enter wearable sensor readings*")
    st.divider()

    st.markdown("### 📱 Accelerometer")
    acc_x = st.slider("Acceleration X", -6.0, 6.0, 0.265, 0.001)
    acc_y = st.slider("Acceleration Y", -4.0, 4.0, -0.781, 0.001)
    acc_z = st.slider("Acceleration Z", -4.0, 4.0, -0.007, 0.001)

    st.divider()

    st.markdown("### 🔄 Gyroscope")
    gyro_x = st.slider("Gyroscope X", -5.0, 5.0, -0.059, 0.001)
    gyro_y = st.slider("Gyroscope Y", -8.0, 9.0, 0.032, 0.001)
    gyro_z = st.slider("Gyroscope Z", -10.0, 12.0, -2.929, 0.001)

    st.divider()

    st.markdown("### ⌚ Wrist")
    wrist = st.radio("Which wrist?", ["Left (0)", "Right (1)"])
    wrist_val = 0 if "Left" in wrist else 1

    st.divider()
    st.caption("Built with Scikit-learn + Streamlit")

# Calculate magnitude features automatically
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

# Main content - two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🔢 Current Sensor Readings")
    readings = pd.DataFrame({
        "Feature": [
            "Wrist",
            "Acceleration X",
            "Acceleration Y",
            "Acceleration Z",
            "Gyroscope X",
            "Gyroscope Y",
            "Gyroscope Z",
            "Acc Magnitude",
            "Gyro Magnitude"
        ],
        "Value": [
            wrist_val,
            acc_x, acc_y, acc_z,
            gyro_x, gyro_y, gyro_z,
            round(acc_magnitude, 4),
            round(gyro_magnitude, 4)
        ]
    })
    st.dataframe(readings, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 🎯 Prediction")
    st.markdown("")

    if st.button("🔍 Predict Activity"):
        # Prepare raw input with all 9 features
        raw_input = np.array([[
            wrist_val,
            acc_x, acc_y, acc_z,
            gyro_x, gyro_y, gyro_z,
            acc_magnitude, gyro_magnitude
        ]])

        # Scale input using saved scaler
        scaled_input = scaler.transform(raw_input)

        # Predict
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]

        st.markdown("")

        if prediction == 0:
            st.success("## 🚶 Walking")
            st.markdown(f"**Confidence: {probability[0]*100:.2f}%**")
            st.progress(float(probability[0]))
        else:
            st.error("## 🏃 Running")
            st.markdown(f"**Confidence: {probability[1]*100:.2f}%**")
            st.progress(float(probability[1]))

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🚶 Walk Probability", f"{probability[0]*100:.2f}%")
        with col_b:
            st.metric("🏃 Run Probability", f"{probability[1]*100:.2f}%")

st.divider()

# Model comparison section
st.markdown("### 📊 Model Performance Comparison")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN (k=5)",
        "MLP Neural Network ⭐",
        "Random Forest (Tuned)",
        "KNN (Tuned)"
    ],
    "Test Accuracy": [
        "95.33%", "98.25%", "98.99%",
        "99.16%", "99.24%", "99.01%", "99.16%"
    ],
    "Notes": [
        "Linear baseline",
        "No scaling needed",
        "100 trees ensemble",
        "Distance based",
        "Best model — deployed",
        "GridSearchCV tuned",
        "GridSearchCV tuned"
    ]
})

st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()

# Pipeline
st.markdown("### 🏗️ Pipeline")
st.markdown("""""")
st.divider()
st.caption("Human Activity Recognition | MLP Neural Network | 99.24% Accuracy | Built with Scikit-learn + Streamlit")