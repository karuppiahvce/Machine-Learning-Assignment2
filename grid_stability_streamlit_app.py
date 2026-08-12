import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Electrical Grid Stability Classification",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electrical Grid Stability Classification")
st.markdown(
    "Machine Learning classification using Logistic Regression, "
    "Decision Tree, KNN, Gaussian Naive Bayes and Random Forest."
)

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]

TARGET = "stabf"

# ---------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------
st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Grid stability test dataset.csv",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    default_file = "Grid stability test dataset.csv"
    try:
        data = pd.read_csv(default_file)
        st.sidebar.success(f"Loaded: {default_file}")
    except FileNotFoundError:
        st.warning(
            "Please upload the dataset CSV file using the sidebar. "
            "The notebook expects a file named 'Grid stability test dataset.csv'."
        )
        st.stop()

# ---------------------------------------------------------
# VALIDATE DATASET
# ---------------------------------------------------------
required_columns = FEATURES + [TARGET]
missing_columns = [c for c in required_columns if c not in data.columns]

if missing_columns:
    st.error(
        "The uploaded dataset is missing the following required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# ---------------------------------------------------------
# CREATE TRAIN / TEST DATA
# Same split as the supplied notebook
# ---------------------------------------------------------
train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data[TARGET]
)

# ---------------------------------------------------------
# SIDEBAR INFORMATION
# ---------------------------------------------------------
st.sidebar.subheader("Dataset Information")
st.sidebar.write(f"Original rows: **{data.shape[0]}**")
st.sidebar.write(f"Original columns: **{data.shape[1]}**")
st.sidebar.write(f"Training rows: **{train_data.shape[0]}**")
st.sidebar.write(f"Test rows: **{test_data.shape[0]}**")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset",
    "🤖 Model Results",
    "🏆 Best Model",
    "🔮 Prediction",
    "⬇️ Downloads"
])

# ---------------------------------------------------------
# TAB 1 - DATASET
# ---------------------------------------------------------
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Original Dataset", f"{len(data):,} rows")

    with col2:
        st.metric("Training Dataset", f"{len(train_data):,} rows")

    with col3:
        st.metric("Test Dataset", f"{len(test_data):,} rows")

    st.subheader("Original Dataset - First 10 Rows")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Training Dataset")
    st.write(f"Shape: {train_data.shape}")
    st.dataframe(train_data, use_container_width=True, height=400)

    st.subheader("Test Dataset")
    st.write(f"Shape: {test_data.shape}")
    st.dataframe(test_data, use_container_width=True, height=400)

    st.subheader("Missing Values")
    missing = data.isnull().sum().to_frame("Missing Values")
    st.dataframe(missing, use_container_width=True)

    st.subheader("Target Distribution")
    st.dataframe(
        data[TARGET].value_counts().rename_axis(TARGET).reset_index(name="Count"),
        use_container_width=True
    )

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
encoder = LabelEncoder()

y_train_raw = train_data[TARGET]
y_test_raw = test_data[TARGET]

y_train = encoder.fit_transform(y_train_raw)
y_test = encoder.transform(y_test_raw)

X_train = train_data[FEATURES]
X_test = test_data[FEATURES]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        X_train_scaled,
        X_test_scaled
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        X_train,
        X_test
    ),
    "K-Nearest Neighbour": (
        KNeighborsClassifier(n_neighbors=5),
        X_train_scaled,
        X_test_scaled
    ),
    "Gaussian Naive Bayes": (
        GaussianNB(),
        X_train,
        X_test
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train,
        X_test
    )
}

model_objects = {}
predictions = {}
probabilities = {}
result_rows = []

for name, (model, train_x, test_x) in models.items():
    model.fit(train_x, y_train)

    pred = model.predict(test_x)
    prob = model.predict_proba(test_x)[:, 1]

    model_objects[name] = model
    predictions[name] = pred
    probabilities[name] = prob

    result_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC Score": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, pred)
    })

results = pd.DataFrame(result_rows).round(4)

best_index = results["F1 Score"].idxmax()
best_model = results.loc[best_index, "Model"]

# ---------------------------------------------------------
# TAB 2 - MODEL RESULTS
# ---------------------------------------------------------
with tab2:
    st.header("Final Comparison of Five Machine Learning Models")

    st.dataframe(
        results.style.highlight_max(
            subset=["Accuracy", "AUC Score", "Precision", "Recall",
                    "F1 Score", "MCC Score"],
            color="#d9f2d9"
        ),
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")
    chart_data = results.set_index("Model")[["Accuracy"]]
    st.bar_chart(chart_data)

    st.subheader("F1 Score Comparison")
    f1_data = results.set_index("Model")[["F1 Score"]]
    st.bar_chart(f1_data)

# ---------------------------------------------------------
# TAB 3 - BEST MODEL
# ---------------------------------------------------------
with tab3:
    st.header("🏆 Best Performing Model")

    best_row = results.loc[best_index]

    st.success(f"Best Model based on F1 Score: **{best_model}**")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        st.metric("Accuracy", best_row["Accuracy"])
    with c2:
        st.metric("AUC Score", best_row["AUC Score"])
    with c3:
        st.metric("Precision", best_row["Precision"])
    with c4:
        st.metric("Recall", best_row["Recall"])
    with c5:
        st.metric("F1 Score", best_row["F1 Score"])
    with c6:
        st.metric("MCC Score", best_row["MCC Score"])

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions[best_model])
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in encoder.classes_],
        columns=[f"Predicted {c}" for c in encoder.classes_]
    )
    st.dataframe(cm_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 4 - NEW SAMPLE PREDICTION
# ---------------------------------------------------------
with tab4:
    st.header("🔮 Predict Grid Stability for a New Sample")

    st.write("Enter values for the 12 input features.")

    default_values = {
        "tau1": 2.0,
        "tau2": 1.0,
        "tau3": 0.0,
        "tau4": 0.5,
        "p1": 3.0,
        "p2": -1.0,
        "p3": -1.2,
        "p4": -0.8,
        "g1": 0.6,
        "g2": 0.7,
        "g3": 0.8,
        "g4": 0.9
    }

    input_values = {}

    cols = st.columns(3)

    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            input_values[feature] = st.number_input(
                feature,
                value=float(default_values[feature]),
                format="%.6f"
            )

    if st.button("Predict Stability", type="primary"):
        new_sample = pd.DataFrame([input_values])[FEATURES]
        new_sample_scaled = scaler.transform(new_sample)

        prediction_results = []

        for name, model in model_objects.items():
            if name in ["Logistic Regression", "K-Nearest Neighbour"]:
                prediction = model.predict(new_sample_scaled)[0]
                probability = model.predict_proba(new_sample_scaled)[0, 1]
            else:
                prediction = model.predict(new_sample)[0]
                probability = model.predict_proba(new_sample)[0, 1]

            label = encoder.inverse_transform([prediction])[0]

            prediction_results.append({
                "Model": name,
                "Prediction": label,
                "Probability of Class 1": round(float(probability), 4)
            })

        prediction_df = pd.DataFrame(prediction_results)

        st.subheader("Prediction Results")
        st.dataframe(prediction_df, use_container_width=True)

        best_prediction = prediction_df.loc[
            prediction_df["Model"] == best_model, "Prediction"
        ].iloc[0]

        st.success(
            f"**{best_model}** predicts the grid as: **{best_prediction}**"
        )

# ---------------------------------------------------------
# TAB 5 - DOWNLOADS
# ---------------------------------------------------------
with tab5:
    st.header("⬇️ Download Datasets and Results")

    original_csv = data.to_csv(index=False).encode("utf-8")
    train_csv = train_data.to_csv(index=False).encode("utf-8")
    test_csv = test_data.to_csv(index=False).encode("utf-8")
    results_csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Original Dataset",
        original_csv,
        "Grid_Stability_Original.csv",
        "text/csv"
    )

    st.download_button(
        "Download Training Dataset",
        train_csv,
        "Grid_Stability_Training.csv",
        "text/csv"
    )

    st.download_button(
        "Download Test Dataset",
        test_csv,
        "Grid_Stability_Test.csv",
        "text/csv"
    )

    st.download_button(
        "Download ML Results",
        results_csv,
        "Grid_Stability_ML_Results.csv",
        "text/csv"
    )

    st.info(
        "The train/test split follows the supplied notebook: "
        "80% training, 20% testing, random_state=42, stratified by stabf."
    )
