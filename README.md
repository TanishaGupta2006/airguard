# AirGuard - Urban Air Quality Risk Predictor

A simple, ML-centric college project that follows the machine-learning lifecycle from data preparation to model evaluation and a Streamlit frontend.

## Project objective

Predict the **AQI category** from pollutant measurements for Indian cities.

Target classes:
- Good
- Satisfactory
- Moderate
- Poor
- Very Poor
- Severe

## Dataset

Recommended dataset: **Air Quality Data in India (2015–2020)** by Rohan Rao/Vopani on Kaggle. It contains hourly/daily air-quality and AQI data from Indian cities and includes `AQI_Bucket`. The original data was made publicly available by the Central Pollution Control Board (CPCB).

Kaggle: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Download `city_day.csv` and place it here:

```text
data/city_day.csv
```

## Tech stack

- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn (available for extension)
- Joblib
- VS Code

## Run in VS Code

### 1. Create and activate virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add dataset

Download `city_day.csv` from the Kaggle link above and put it inside `data/`.

### 4. Train the models

```bash
python src/train_model.py
```

This compares Logistic Regression and Random Forest and saves the best model to `models/`.

### 5. Start the frontend

```bash
streamlit run app.py
```

## ML lifecycle covered

```text
Data Collection
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Cross Validation
      ↓
Evaluation
      ↓
Best Model
      ↓
Streamlit Prediction App
```

## Models

### Logistic Regression
Used as the baseline classification model.

### Random Forest
Used as the stronger nonlinear model and selected automatically if it performs better on the held-out test set.

## Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- 5-fold cross-validation F1
- Classification report
- Confusion matrix saved during training