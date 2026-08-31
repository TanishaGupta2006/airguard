import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "city_day.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        "Dataset not found. Download city_day.csv from the Kaggle dataset and place it in data/city_day.csv"
    )

df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]

required = ["City", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "AQI_Bucket"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Expected the city_day.csv file from the selected dataset.")

features = ["City", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
target = "AQI_Bucket"

work = df[features + [target]].copy()
work[target] = work[target].astype(str).str.strip().str.lower()
work = work[work[target].isin(["good", "satisfactory", "moderate", "poor", "very poor", "severe"])]

# Remove rows with no pollutant information at all.
pollutants = [c for c in features if c != "City"]
work = work.dropna(subset=pollutants, how="all")

X = work[features]
y = work[target]

cat_cols = ["City"]
num_cols = pollutants

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1))
    ])
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }

# Cross-validation on the two candidates.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    results[name]["cv_f1_mean"] = float(scores.mean())
    results[name]["cv_f1_std"] = float(scores.std())

best_name = max(results, key=lambda n: results[n]["f1"])
best_model = models[best_name]
best_model.fit(X, y)

joblib.dump(best_model, MODELS / "airguard_model.joblib")
with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"best_model": best_name, "results": results, "classes": sorted(y.unique().tolist()), "rows_used": int(len(work))}, f, indent=2)

# Save a small test report for the project demo.
pred = best_model.predict(X_test)
report = classification_report(y_test, pred, output_dict=True, zero_division=0)
with open(MODELS / "classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
np.save(MODELS / "confusion_matrix.npy", confusion_matrix(y_test, pred, labels=sorted(y.unique())))

print(f"Rows used: {len(work)}")
print(f"Best model: {best_name}")
print(pd.DataFrame(results).T.round(4))
print("Saved model to models/airguard_model.joblib")
