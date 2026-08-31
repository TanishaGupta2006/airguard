from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "airguard_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="AirGuard",layout="wide")

st.markdown("# AirGuard")
st.caption("Urban Air Quality Risk Prediction using Machine Learning")

if not MODEL_PATH.exists():
    st.error("Model is not trained yet. Run: python src/train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

cities = ["Ahmedabad", "Aizawl", "Amaravati", "Amritsar", "Bengaluru", "Bhopal", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Patna", "Visakhapatnam"]

risk_info = {
    "good": ("", "Good", "Air quality is considered satisfactory."),
    "satisfactory": ("", "Satisfactory", "Air quality is acceptable for most people."),
    "moderate": ("", "Moderate", "Sensitive people may experience health effects."),
    "poor": ("", "Poor", "Health effects are possible, especially for sensitive groups."),
    "very poor": ("", "Very Poor", "Prolonged exposure may cause significant health effects."),
    "severe": ("", "Severe", "Health alert: avoid prolonged outdoor exposure."),
}

st.sidebar.header("Air Quality Inputs")
city = st.sidebar.selectbox("City", cities, index=cities.index("Chennai"))

cols = st.sidebar.columns(2)
def num(label, value, min_value=0.0, max_value=500.0, step=1.0):
    return st.sidebar.number_input(
        label,
        value=float(value),
        min_value=float(min_value),
        max_value=float(max_value),
        step=float(step),
    )

pm25 = num("PM2.5 (µg/m³)", 65.0, 0.0, 500.0, 1.0)

pm10 = num("PM10 (µg/m³)", 110.0, 0.0, 600.0, 1.0)

no = num("NO (µg/m³)", 20.0, 0.0, 500.0, 1.0)

no2 = num("NO₂ (µg/m³)", 40.0, 0.0, 500.0, 1.0)

nox = num("NOx (µg/m³)", 50.0, 0.0, 500.0, 1.0)

nh3 = num("NH₃ (µg/m³)", 20.0, 0.0, 500.0, 1.0)

co = num("CO (mg/m³)", 0.8, 0.0, 20.0, 0.1)

so2 = num("SO₂ (µg/m³)", 18.0, 0.0, 500.0, 1.0)

o3 = num("O₃ (µg/m³)", 55.0, 0.0, 500.0, 1.0)

input_df = pd.DataFrame([{
    "City": city, "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2,
    "NOx": nox, "NH3": nh3, "CO": co, "SO2": so2, "O3": o3
}])

if st.sidebar.button("Predict Air Quality", type="primary", use_container_width=True):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
    icon, label, explanation = risk_info.get(prediction, ("", prediction.title(), "Model prediction."))

    st.subheader("Prediction")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted Risk", f"{icon} {label}")
    with c2:
        if proba is not None:
            confidence = float(np.max(proba)) * 100
            st.metric("Model Confidence", f"{confidence:.1f}%")
        st.info(explanation)

    if proba is not None:
        classes = list(model.classes_)
        chart = pd.DataFrame({"Category": classes, "Probability": proba}).set_index("Category")
        st.subheader("Prediction Probability")
        st.bar_chart(chart)

st.divider()

st.subheader("Project Model Performance")
if metrics:
    st.write(f"**Selected model:** {metrics.get('best_model', 'N/A')}")
    rows = []
    for name, values in metrics.get("results", {}).items():
        rows.append({
            "Model": name,
            "Accuracy": round(values["accuracy"] * 100, 2),
            "Precision": round(values["precision"] * 100, 2),
            "Recall": round(values["recall"] * 100, 2),
            "F1 Score": round(values["f1"] * 100, 2),
            "5-Fold CV F1": round(values["cv_f1_mean"] * 100, 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("About this project"):
    st.markdown("""
    **ML lifecycle implemented:**
    1. Data collection from a public Indian air-quality dataset.
    2. Data cleaning and missing-value handling.
    3. Exploratory analysis and model comparison.
    4. Feature preprocessing: median imputation, scaling and one-hot encoding.
    5. Supervised classification using Logistic Regression and Random Forest.
    6. Cross-validation and evaluation using Accuracy, Precision, Recall and F1-score.
    7. Best model saved with Joblib and used by this Streamlit interface.
    """)