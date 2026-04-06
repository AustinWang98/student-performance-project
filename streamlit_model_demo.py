import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Anomaly Detection", layout="wide")

st.title("Student Performance Anomaly Detection")

st.write(
"""
Interactive demo using **Isolation Forest** to detect anomalous student profiles
based on behavioral and academic attributes.
"""
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("iforest_model.pkl")

model = load_model()

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():

    mat = pd.read_csv("student/student-mat.csv", sep=";")
    mat["subject"] = "math"

    por = pd.read_csv("student/student-por.csv", sep=";")
    por["subject"] = "portuguese"

    df = pd.concat([mat, por], ignore_index=True)

    df = df.drop(columns=["G1", "G2"])

    return df

df = load_data()

# -----------------------------
# Dataset overview
# -----------------------------
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))
col2.metric("Features", df.shape[1])
col3.metric("Model", "Isolation Forest")

st.dataframe(df.head())

# -----------------------------
# Model prediction on dataset
# -----------------------------
X = df.copy()

scores = -model.decision_function(X)
preds = model.predict(X)

df["anomaly_score"] = scores
df["anomaly"] = preds

# -----------------------------
# Score distribution
# -----------------------------
st.header("Anomaly Score Distribution")

fig, ax = plt.subplots()

ax.hist(scores, bins=40)

ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Anomaly Scores")

st.pyplot(fig)

# -----------------------------
# Top anomalies
# -----------------------------
st.header("Most Anomalous Students")

top_anomalies = df.sort_values("anomaly_score", ascending=False).head(10)

st.dataframe(top_anomalies)

# -----------------------------
# Interactive student check
# -----------------------------
st.header("Interactive Student Check")

st.write("Modify key attributes and detect whether the student profile is anomalous.")

age = st.slider("Age", 15, 22, 17)
studytime = st.slider("Study time", 1, 4, 2)
failures = st.slider("Past failures", 0, 4, 0)
absences = st.slider("Absences", 0, 50, 5)
goout = st.slider("Going out frequency", 1, 5, 3)
freetime = st.slider("Free time", 1, 5, 3)

if st.button("Detect anomaly"):

    # Use template student
    sample = df.iloc[[0]].copy()

    sample["age"] = age
    sample["studytime"] = studytime
    sample["failures"] = failures
    sample["absences"] = absences
    sample["goout"] = goout
    sample["freetime"] = freetime

    score = -model.decision_function(sample)[0]
    pred = model.predict(sample)[0]

    st.subheader("Prediction Result")

    st.write("Anomaly score:", round(score, 4))

    if pred == -1:
        st.error("This student profile appears anomalous.")
    else:
        st.success("This student profile looks normal.")

# -----------------------------
# Scatter visualization
# -----------------------------
st.header("Absences vs Failures (Anomaly Visualization)")

fig2, ax2 = plt.subplots()

size = df["anomaly_score"] * 200

scatter = ax2.scatter(
    df["absences"],
    df["failures"],
    s=size,
    c=df["anomaly_score"],
    cmap="coolwarm",
    alpha=0.6
)

plt.colorbar(scatter, label="Anomaly Score")

ax2.set_xlabel("Absences")
ax2.set_ylabel("Failures")
ax2.set_title("Student Anomaly Detection")

st.pyplot(fig2)

# -----------------------------
# Behavior comparison
# -----------------------------
st.header("Behavior Pattern Comparison: Normal vs Anomaly")

features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "goout",
    "freetime",
    "health",
    "Dalc",
    "Walc",
    "Medu"
]

normal = df[df["anomaly"] == 1]
anomaly = df[df["anomaly"] == -1]

normal_mean = normal[features].mean()
anomaly_mean = anomaly[features].mean()

fig3, ax3 = plt.subplots()

x = np.arange(len(features))
width = 0.35

ax3.bar(x - width/2, normal_mean, width, label="Normal")
ax3.bar(x + width/2, anomaly_mean, width, label="Anomaly")

ax3.set_xticks(x)
ax3.set_xticklabels(features, rotation=45)

ax3.set_ylabel("Average Value")
ax3.set_title("Behavioral Feature Comparison")

ax3.legend()

st.pyplot(fig3)