import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------
# 1 Load datasets
# ---------------------------

mat_path = "student/student-mat.csv"
por_path = "student/student-por.csv"

df_mat = pd.read_csv(mat_path, sep=";")
df_mat["subject"] = "mat"

df_por = pd.read_csv(por_path, sep=";")
df_por["subject"] = "por"

df = pd.concat([df_mat, df_por], axis=0, ignore_index=True)

print("Dataset shape:", df.shape)

# ---------------------------
# 2 Remove leakage variables
# ---------------------------

DROP_COLS = ["G1", "G2"]

df_model = df.drop(columns=DROP_COLS).copy()
df_model = df_model.drop_duplicates().reset_index(drop=True)

print("After dropping G1/G2:", df_model.shape)

# ---------------------------
# 3 Identify feature types
# ---------------------------

cat_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
num_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

X = df_model[cat_cols + num_cols].copy()

print("Categorical features:", cat_cols)
print("Numeric features:", num_cols)

# ---------------------------
# 4 Preprocessing
# ---------------------------

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# ---------------------------
# 5 Train/Test split
# ---------------------------

X_train, X_test = train_test_split(
    X,
    test_size=0.30,
    random_state=RANDOM_STATE
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ---------------------------
# 6 Isolation Forest model
# ---------------------------

base_contamination = 0.08

pipe = Pipeline(
    steps=[
        ("prep", preprocess),
        ("iforest", IsolationForest(
            n_estimators=200,
            contamination=base_contamination,
            max_samples="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ]
)

# ---------------------------
# 7 Fit model
# ---------------------------

pipe.fit(X_train)

print("Model training complete")

# ---------------------------
# 8 Evaluate anomaly rate
# ---------------------------

pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)

train_outlier = (pred_train == -1).astype(int)
test_outlier = (pred_test == -1).astype(int)

print("Predicted outlier rate (train):", train_outlier.mean())
print("Predicted outlier rate (test):", test_outlier.mean())

# ---------------------------
# 9 Save trained model
# ---------------------------

joblib.dump(pipe, "iforest_model.pkl")

print("Model saved as iforest_model.pkl")