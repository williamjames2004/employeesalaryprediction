
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data.csv"
MODEL_PATH = "employee_salary_model.pkl"

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

features = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
]
target = "Salary"

required_columns = features + [target]
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"Missing columns in data.csv: {missing_columns}")

data = data[required_columns].copy()

data["Salary"] = pd.to_numeric(data["Salary"], errors="coerce")
data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
data["Years of Experience"] = pd.to_numeric(
    data["Years of Experience"], errors="coerce"
)

data = data.dropna(subset=["Salary"])

if len(data) < 2:
    raise ValueError("At least 2 valid salary records are required.")

X = data[features]
y = data[target]

categorical_features = ["Gender", "Education Level", "Job Title"]
numerical_features = ["Age", "Years of Experience"]

preprocessor = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features),
    ("numerical", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numerical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="squared_error",
        random_state=42
    ))
])

if len(data) >= 10 and y.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Model Evaluation")
    print("Mean Absolute Error:", round(
        mean_absolute_error(y_test, predictions), 2
    ))

    if y_test.nunique() > 1:
        print("R2 Score:", round(r2_score(y_test, predictions), 4))
    else:
        print("R2 Score skipped because the test target has one unique value.")
else:
    print("Dataset is small. Evaluation skipped.")

model.fit(X, y)
joblib.dump(model, MODEL_PATH)

print("Employee salary model trained and saved successfully!")
print("Training records:", len(data))
print("Algorithm: Gradient Boosting Regressor")