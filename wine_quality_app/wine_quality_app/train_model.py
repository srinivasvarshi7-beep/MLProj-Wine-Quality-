"""
train_model.py  -  run ONCE to create the model files used by the website.

It repeats the exact steps of your notebook for the final model (Tuned XGBoost):
  1. load WineQT.csv and drop the 'Id' column
  2. 80/20 stratified split (random_state=42)
  3. LabelEncoder on quality (3..8  ->  0..5)
  4. XGBoost with the best hyper-parameters found by RandomizedSearchCV in the notebook
It then saves:
  models/wine_quality_model.json   (XGBoost native format - safest to load)
  models/model_metadata.json       (feature order, label mapping, defaults, test metrics, etc.)

Run:  python train_model.py
"""
import json
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

DATA_PATH = "data/WineQT.csv"
MODEL_PATH = "models/wine_quality_model.json"
META_PATH = "models/model_metadata.json"

# Best parameters printed by RandomizedSearchCV in the notebook (cell 29 / cell 48)
BEST_PARAMS = {"subsample": 0.8, "n_estimators": 100, "min_child_weight": 1,
               "max_depth": 5, "learning_rate": 0.1, "colsample_bytree": 0.7}

df = pd.read_csv(DATA_PATH).drop("Id", axis=1)       # Id is only an identifier
X = df.drop("quality", axis=1)
y = df["quality"]
FEATURES = list(X.columns)                           # exact training feature order

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

model = XGBClassifier(objective="multi:softmax", num_class=6, random_state=42, **BEST_PARAMS)
model.fit(X_train, y_train_enc)

pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test_enc, pred) * 100,
    "precision_weighted": precision_score(y_test_enc, pred, average="weighted", zero_division=0) * 100,
    "recall_weighted": recall_score(y_test_enc, pred, average="weighted", zero_division=0) * 100,
    "f1_weighted": f1_score(y_test_enc, pred, average="weighted", zero_division=0) * 100,
    "f1_macro": f1_score(y_test_enc, pred, average="macro", zero_division=0) * 100,
}
print("Test metrics (should match the notebook: 69.87 / 67.23 / 69.87 / 68.32 / 41.12):")
for k, v in metrics.items():
    print(f"  {k:20s} {v:.2f}")

labels = np.arange(len(le.classes_))
report = classification_report(y_test_enc, pred, labels=labels,
                               target_names=[str(c) for c in le.classes_],
                               zero_division=0, output_dict=True)
cm = confusion_matrix(y_test_enc, pred, labels=labels)

# Save in XGBoost's native JSON format
model.save_model(MODEL_PATH)

meta = {
    "features": FEATURES,
    "classes": [int(c) for c in le.classes_],            # index i -> quality score
    "best_params": BEST_PARAMS,
    "feature_stats": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                          "median": float(df[c].median()),
                          "p01": float(df[c].quantile(0.01)), "p99": float(df[c].quantile(0.99))}
                      for c in FEATURES},
    "test_metrics": metrics,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "feature_importance": dict(zip(FEATURES, [float(v) for v in model.feature_importances_])),
    "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
              "test_size": 0.20, "random_state": 42, "stratified": True},
    "versions": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__},
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved {MODEL_PATH} and {META_PATH}")
