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
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ============================================================
# ELECTRICAL GRID STABILITY - STREAMLIT APP
# ============================================================
# Keep this file and Grid_Stability_Training.csv in the same
# folder. The user uploads ONLY Grid_Stability_Test.csv.
#
# When "All Models" is selected:
#   - Metrics for all five models are displayed
#   - Confusion matrices for all five models are displayed
#
# When one model is selected:
#   - Metrics for that model are displayed
#   - Confusion matrix for that model is displayed
# ============================================================

st.set_page_config(
    page_title="Electrical Grid Stability ML",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electrical Grid Stability Classification")
st.write(
    "Upload the test dataset and evaluate the five machine learning models."
)

# ------------------------------------------------------------
# Feature names from the assignment notebook
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Train all models using the training CSV
# ------------------------------------------------------------

@st.cache_resource
def train_models():

    training_file = "Grid_Stability_Training.csv"

    train_data = pd.read_csv(training_file)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in train_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training dataset is missing: "
            + ", ".join(missing_columns)
        )

    X_train = train_data[FEATURES]
    y_train_text = train_data[TARGET]

    # Convert stable/unstable to 0/1
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)

    # Scale training data for Logistic Regression and KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "K-Nearest Neighbour": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Gaussian Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # Train Logistic Regression and KNN with scaled data
    models["Logistic Regression"].fit(
        X_train_scaled,
        y_train
    )

    models["K-Nearest Neighbour"].fit(
        X_train_scaled,
        y_train
    )

    # Train the remaining models with original features
    models["Decision Tree"].fit(
        X_train,
        y_train
    )

    models["Gaussian Naive Bayes"].fit(
        X_train,
        y_train
    )

    models["Random Forest"].fit(
        X_train,
        y_train
    )

    return models, scaler, encoder


# ------------------------------------------------------------
# Load the trained models
# ------------------------------------------------------------

try:
    models, scaler, encoder = train_models()

except Exception as error:
    st.error(
        "Training dataset could not be loaded."
    )

    st.error(
        "Make sure Grid_Stability_Training.csv is in the "
        "same folder as this Python file."
    )

    st.error(str(error))

    st.stop()


# ------------------------------------------------------------
# Upload TEST dataset only
# ------------------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload Grid_Stability_Test.csv",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload the test CSV file to start the evaluation."
    )

    st.markdown(
        """
        **Expected columns**

        tau1, tau2, tau3, tau4, p1, p2, p3, p4,
        g1, g2, g3, g4, stabf
        """
    )

    st.stop()


# ------------------------------------------------------------
# Read uploaded test dataset
# ------------------------------------------------------------

test_data = pd.read_csv(uploaded_file)

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in test_data.columns
]

if missing_columns:

    st.error(
        "The uploaded test dataset is missing: "
        + ", ".join(missing_columns)
    )

    st.stop()


st.success("Test dataset uploaded successfully!")

st.write(
    "Test dataset size:",
    test_data.shape
)

# Show uploaded data
with st.expander("View Uploaded Test Dataset"):

    st.dataframe(
        test_data,
        use_container_width=True
    )


# ------------------------------------------------------------
# Prepare test data
# ------------------------------------------------------------

X_test = test_data[FEATURES]

try:
    y_test = encoder.transform(
        test_data[TARGET]
    )

except ValueError:

    st.error(
        "The test dataset contains target labels "
        "different from the training dataset."
    )

    st.write(
        "Expected labels:",
        list(encoder.classes_)
    )

    st.stop()


# ------------------------------------------------------------
# Model selection
# ------------------------------------------------------------

st.header("2. Model Selection")

selected_model = st.selectbox(
    "Select a model:",
    ["All Models"] + MODEL_NAMES
)


# ------------------------------------------------------------
# Function to evaluate one model
# ------------------------------------------------------------

def evaluate_model(model_name):

    model = models[model_name]

    # Logistic Regression and KNN use scaled features
    if model_name in [
        "Logistic Regression",
        "K-Nearest Neighbour"
    ]:

        X_input = scaler.transform(
            X_test
        )

    else:

        X_input = X_test

    prediction = model.predict(
        X_input
    )

    probability = model.predict_proba(
        X_input
    )[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            prediction
        ),

        "AUC Score": roc_auc_score(
            y_test,
            probability
        ),

        "Precision": precision_score(
            y_test,
            prediction,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            prediction,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            prediction,
            zero_division=0
        ),

        "MCC Score": matthews_corrcoef(
            y_test,
            prediction
        )
    }

    return prediction, metrics


# ------------------------------------------------------------
# Function to display one confusion matrix
# ------------------------------------------------------------

def show_confusion_matrix(
    model_name,
    prediction
):

    cm = confusion_matrix(
        y_test,
        prediction
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.imshow(cm)

    ax.set_title(
        "Confusion Matrix\n" + model_name
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        encoder.classes_
    )

    ax.set_yticklabels(
        encoder.classes_
    )

    # Write values inside the matrix
    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)


# ============================================================
# ALL MODELS
# ============================================================

if selected_model == "All Models":

    st.header("3. Evaluation Metrics - All Models")

    result_rows = []
    predictions = {}

    for model_name in MODEL_NAMES:

        prediction, metrics = evaluate_model(
            model_name
        )

        predictions[model_name] = prediction

        result_rows.append({
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "AUC Score": metrics["AUC Score"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "MCC Score": metrics["MCC Score"]
        })

    results = pd.DataFrame(
        result_rows
    ).round(4)

    # Best model based on F1 Score
    best_index = results[
        "F1 Score"
    ].idxmax()

    best_model = results.loc[
        best_index,
        "Model"
    ]

    # Highlight best model
    def highlight_best(row):

        if row.name == best_index:

            return [
                "background-color: lightgreen"
            ] * len(row)

        return [
            ""
        ] * len(row)

    st.dataframe(
        results.style.apply(
            highlight_best,
            axis=1
        ),
        use_container_width=True
    )

    st.success(
        "Best model based on F1 Score: "
        + best_model
    )

    # --------------------------------------------------------
    # Confusion matrices for ALL FIVE models
    # --------------------------------------------------------

    st.header(
        "4. Confusion Matrices - All Five Models"
    )

    # Display two matrices per row
    for start in range(
        0,
        len(MODEL_NAMES),
        2
    ):

        columns = st.columns(2)

        for position in range(2):

            index = start + position

            if index >= len(MODEL_NAMES):
                break

            model_name = MODEL_NAMES[index]

            with columns[position]:

                st.subheader(
                    model_name
                )

                show_confusion_matrix(
                    model_name,
                    predictions[model_name]
                )

    # --------------------------------------------------------
    # Classification report for best model
    # --------------------------------------------------------

    st.header(
        "5. Classification Report - Best Model"
    )

    best_prediction = predictions[
        best_model
    ]

    report = classification_report(
        y_test,
        best_prediction,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose().round(4)

    st.dataframe(
        report_df,
        use_container_width=True
    )


# ============================================================
# PARTICULAR MODEL
# ============================================================

else:

    st.header(
        "3. Evaluation Results - "
        + selected_model
    )

    prediction, metrics = evaluate_model(
        selected_model
    )

    # Display six metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        round(metrics["Accuracy"], 4)
    )

    col2.metric(
        "AUC Score",
        round(metrics["AUC Score"], 4)
    )

    col3.metric(
        "Precision",
        round(metrics["Precision"], 4)
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        round(metrics["Recall"], 4)
    )

    col5.metric(
        "F1 Score",
        round(metrics["F1 Score"], 4)
    )

    col6.metric(
        "MCC Score",
        round(metrics["MCC Score"], 4)
    )

    # --------------------------------------------------------
    # Confusion matrix for selected model ONLY
    # --------------------------------------------------------

    st.header(
        "4. Confusion Matrix - "
        + selected_model
    )

    show_confusion_matrix(
        selected_model,
        prediction
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    st.header(
        "5. Classification Report"
    )

    report = classification_report(
        y_test,
        prediction,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose().round(4)

    st.dataframe(
        report_df,
        use_container_width=True
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "Electrical Grid Stability Classification | "
    "Logistic Regression | Decision Tree | KNN | "
    "Gaussian Naive Bayes | Random Forest"
)
