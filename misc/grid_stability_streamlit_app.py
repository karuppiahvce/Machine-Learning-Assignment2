import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Electrical Grid Stability", page_icon="⚡", layout="wide")

st.title("⚡ Electrical Grid Stability Prediction")
st.write("Upload ONLY the test dataset and compare five classification models.")
st.info("The training dataset is kept with the application. Only the test CSV needs to be uploaded.")

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]
TARGET = "stabf"

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbour",
    "Gaussian Naive Bayes",
    "Random Forest"
]

@st.cache_resource
def train_models():
    train_data = pd.read_csv("Grid_Stability_Training.csv")

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in train_data.columns]
    if missing:
        raise ValueError("Training file is missing: " + ", ".join(missing))

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbour": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    models["Logistic Regression"].fit(X_train_scaled, y_train)
    models["K-Nearest Neighbour"].fit(X_train_scaled, y_train)

    models["Decision Tree"].fit(X_train, y_train)
    models["Gaussian Naive Bayes"].fit(X_train, y_train)
    models["Random Forest"].fit(X_train, y_train)

    return models, scaler, encoder

try:
    models, scaler, encoder = train_models()
except Exception as e:
    st.error("Training dataset could not be loaded.")
    st.error(str(e))
    st.stop()

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload Grid_Stability_Test.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the test CSV file.")
    st.stop()

test_data = pd.read_csv(uploaded_file)

required = FEATURES + [TARGET]
missing = [c for c in required if c not in test_data.columns]

if missing:
    st.error("Test file is missing: " + ", ".join(missing))
    st.stop()

st.success("Test dataset uploaded successfully!")
st.write("Test dataset size:", test_data.shape)

with st.expander("View Test Dataset"):
    st.dataframe(test_data, use_container_width=True)

X_test = test_data[FEATURES]

try:
    y_test = encoder.transform(test_data[TARGET])
except ValueError:
    st.error("Test target labels do not match the training labels.")
    st.write("Expected labels:", list(encoder.classes_))
    st.stop()

st.header("2. Select Model")

selected_model = st.selectbox(
    "Choose a model:",
    ["All Models"] + MODEL_NAMES
)

def evaluate_model(model_name):
    model = models[model_name]

    if model_name in ["Logistic Regression", "K-Nearest Neighbour"]:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, prediction),
        "AUC Score": roc_auc_score(y_test, probability),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1 Score": f1_score(y_test, prediction, zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, prediction),
        "Prediction": prediction
    }

if selected_model == "All Models":

    st.header("3. Model Comparison")

    rows = []

    for name in MODEL_NAMES:
        r = evaluate_model(name)
        rows.append({
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "AUC Score": r["AUC Score"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1 Score": r["F1 Score"],
            "MCC Score": r["MCC Score"]
        })

    results = pd.DataFrame(rows).round(4)
    best_index = results["F1 Score"].idxmax()
    best_model = results.loc[best_index, "Model"]

    def highlight_best(row):
        if row.name == best_index:
            return ["background-color: lightgreen"] * len(row)
        return [""] * len(row)

    st.dataframe(
        results.style.apply(highlight_best, axis=1),
        use_container_width=True
    )

    st.success("Best model based on F1 Score: " + best_model)

    best = results.loc[best_index]

    st.subheader("Best Model Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", best["Accuracy"])
    c2.metric("AUC Score", best["AUC Score"])
    c3.metric("F1 Score", best["F1 Score"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", best["Precision"])
    c5.metric("Recall", best["Recall"])
    c6.metric("MCC Score", best["MCC Score"])

    prediction = evaluate_model(best_model)["Prediction"]

else:

    st.header("3. Evaluation Results - " + selected_model)

    r = evaluate_model(selected_model)
    prediction = r["Prediction"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", round(r["Accuracy"], 4))
    c2.metric("AUC Score", round(r["AUC Score"], 4))
    c3.metric("Precision", round(r["Precision"], 4))

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", round(r["Recall"], 4))
    c5.metric("F1 Score", round(r["F1 Score"], 4))
    c6.metric("MCC Score", round(r["MCC Score"], 4))

st.subheader("4. Confusion Matrix")

cm = confusion_matrix(y_test, prediction)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("Confusion Matrix")

ax.set_xticks(range(len(encoder.classes_)))
ax.set_yticks(range(len(encoder.classes_)))
ax.set_xticklabels(encoder.classes_)
ax.set_yticklabels(encoder.classes_)

for i in range(len(cm)):
    for j in range(len(cm[i])):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

st.subheader("5. Classification Report")

report = classification_report(
    y_test,
    prediction,
    target_names=encoder.classes_,
    zero_division=0,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(report).transpose().round(4),
    use_container_width=True
)

st.markdown("---")
st.caption("Electrical Grid Stability Classification App")
