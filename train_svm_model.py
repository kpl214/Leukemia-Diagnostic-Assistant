"""
simple vector model
more biased prediction
    - tends to lean towards ~85% chance of no progression or recurrence across many test prompts
    - therefore: hints at potential lack of diversity in training
alternative: see "train_hgb_model.py"
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import joblib

DATA_PATH = "clinical-data/cleaned_leukemia_dataset.csv"
OUT_PATH = "clinical-data/simplified_svm_model.pkl"
df = pd.read_csv(DATA_PATH)

# relevant features
features = [
    'diagnoses.age_at_diagnosis',
    'demographic.gender',
    'demographic.race',
    'diagnoses.primary_diagnosis',
    'diagnoses.progression_or_recurrence',
    'demographic.vital_status'
]
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['diagnoses.age_at_diagnosis']
categorical_features = [
    'demographic.gender',
    'demographic.race',
    'diagnoses.primary_diagnosis',
    'diagnoses.progression_or_recurrence',
    'demographic.vital_status'
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=False)

joblib.dump(model, OUT_PATH)