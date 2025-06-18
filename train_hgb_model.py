"""
saves BOTH the preprocessor and calibrated classifier in one pickle
decision tree -> better for nuanced medical data
    - Results in an AUROC of 0.943
    - More realistic probabilities than SVM
    - Less data leakage/overfitting
    - Better overall performance compared to simpler SVM
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

DATA_PATH = "clinical-data/cleaned_leukemia_dataset.csv"
OUT_PATH  = "clinical-data/hgb_calibrated_bundle.pkl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "diagnoses.age_at_diagnosis",
    "demographic.gender",
    "demographic.race",
    "diagnoses.primary_diagnosis",
    "diagnoses.progression_or_recurrence",
    "demographic.vital_status",
]
df["diagnoses.age_at_diagnosis"] = df["diagnoses.age_at_diagnosis"] / 365.25
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

numeric_features = ["diagnoses.age_at_diagnosis"]
categorical_features = [c for c in features if c not in numeric_features]

numeric_transformer = SimpleImputer(strategy="mean")

# adapting to potentially missing features or information within data
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=5,
            sparse_output=True,
        )),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# preprocessor on training data only
X_train_pre = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out()
print("Final feature names:", list(feature_names))
X_test_pre  = preprocessor.transform(X_test)

# base classifier
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=5,
    l2_regularization=1.0,
    class_weight="balanced",
    random_state=42,
)
hgb.fit(X_train_pre, y_train)


calibrated = CalibratedClassifierCV(
    estimator=hgb,
    method="isotonic",
    cv=5,
    n_jobs=-1,
)

# lots of pipeline/data validation
calibrated.fit(X_train_pre, y_train)
print("classes_ in trained model:", calibrated.classes_)
print(
    "Mean age by label in training set:\n",
    df.groupby("label")["diagnoses.age_at_diagnosis"].mean(),
)


y_pred = calibrated.predict(X_test_pre)
probs   = calibrated.predict_proba(X_test_pre)[:, 1]

print(classification_report(y_test, y_pred))
print("AUROC:", round(roc_auc_score(y_test, probs), 3))

bundle = {"preprocessor": preprocessor, "model": calibrated}
joblib.dump(bundle, OUT_PATH)
print(f"Calibrated model bundle saved to {OUT_PATH}")
