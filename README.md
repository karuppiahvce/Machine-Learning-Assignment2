# Electrical Grid Stability Classification — ML Assignment 2

## a. Problem Statement

Modern electrical grids increasingly rely on decentralized, demand-responsive control rather than fixed supply schedules. The **Decentral Smart Grid Control (DSGC)** concept models a 4-node star network (1 central power producer and 3 consumer/prosumer nodes) and evaluates whether the grid remains **linearly stable** or becomes **unstable** under small perturbations, based on reaction times and power balance of each node.

The objective of this assignment is to formulate this as a **binary classification problem** — predicting whether a given grid configuration is `stable` or `unstable` — and to build, evaluate, and compare multiple supervised machine learning models on this task. The best-performing model is then identified and deployed via an interactive Streamlit web application that allows a user to upload new grid measurements and get real-time stability predictions from all trained models.

## b. Dataset Description

- **Source:** Electrical Grid Stability Simulated Dataset (derived from the DSGC 4-node star network simulation).
- **File used:** `Grid stability test dataset.csv`
- **Total records:** 10,000 rows
- **Total columns:** 14 (12 input features + 2 target-related columns)

**Input Features (12):**

| Feature | Description |
|---|---|
| `tau1`–`tau4` | Reaction time of each of the 4 network participants (1 producer + 3 consumers) |
| `p1`–`p4` | Nominal power produced (positive) or consumed (negative) by each participant |
| `g1`–`g4` | Price elasticity coefficient (gamma) of each participant |

**Target columns:**

- `stab` — a continuous stability measure (maximal real part of the characteristic differential equation root). Not used directly for classification.
- `stabf` — the **categorical target** used for classification: `stable` or `unstable`.

**Class distribution:** `unstable` = 6,380 records, `stable` = 3,620 records (moderately imbalanced, ~64% / 36%).

**Data preparation:**

- No missing values were found in any column.
- The dataset was split into an 80% training set (`Grid_Stability_Training.csv`, 8,000 rows) and a 20% test set (`Grid_Stability_Test.csv`, 2,000 rows) using a stratified split (`random_state=42`) to preserve the class ratio.
- The target `stabf` was label-encoded (`stable` = 0, `unstable` = 1).
- Features were standardized (`StandardScaler`, fit on training data only) for the scale-sensitive models — Logistic Regression and kNN. Tree-based and probabilistic models (Decision Tree, Naive Bayes, Random Forest) used the unscaled features.

## c. GitHub Repository Link

🔗 **[https://github.com/karuppiahvce/Machine-Learning-Assignment2](https://github.com/karuppiahvce/Machine-Learning-Assignment2)**

## d. Models Used

Five supervised classification algorithms were trained on the 12 input features and evaluated on the held-out 20% test set (2,000 records):

1. **Logistic Regression** (scaled features, `max_iter=1000`)
2. **Decision Tree** (unscaled features, `random_state=42`)
3. **k-Nearest Neighbors** (scaled features, `n_neighbors=5`)
4. **Gaussian Naive Bayes** (unscaled features)
5. **Random Forest (Ensemble)** (unscaled features, `n_estimators=100`, `random_state=42`)

### Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8200 | 0.8920 | 0.8408 | 0.8856 | 0.8626 | 0.6039 |
| Decision Tree | 0.8495 | 0.8369 | 0.8818 | 0.8824 | 0.8821 | 0.6741 |
| kNN | 0.8625 | 0.9290 | 0.8547 | 0.9451 | 0.8977 | 0.6980 |
| Naive Bayes | 0.8445 | 0.9210 | 0.8395 | 0.9350 | 0.8847 | 0.6570 |
| Random Forest (Ensemble) | **0.9240** | **0.9806** | **0.9226** | **0.9616** | **0.9417** | **0.8342** |

*(Metrics computed on the 20% stratified test set, with `unstable` treated as the positive class.)*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Gives a reasonable baseline (82% accuracy, AUC 0.89) but it is the weakest of the five models. Since grid stability depends on non-linear interactions between reaction times, power, and elasticity, a purely linear decision boundary struggles to separate the two classes cleanly, which shows up as the lowest MCC (0.60) and F1 (0.86) among all models. |
| **Decision Tree** | Improves over Logistic Regression on Accuracy, Precision, Recall and F1 by capturing non-linear feature interactions and threshold effects. However, its AUC (0.84) is the **lowest of all five models**, lower even than Logistic Regression — a single unpruned tree produces a "blocky", overconfident probability output (fewer probability granularities) and tends to overfit the training data, which hurts ranking quality (AUC) even though hard-label metrics look fine. |
| **kNN** | Performs strongly (86.25% accuracy, AUC 0.929, F1 0.898) once features are standardized. Because the target is essentially governed by local geometric relationships among the scaled features, nearby points in feature space tend to share the same stability label, which suits kNN's local-neighborhood assumption well. It has the highest Recall (0.945) among the non-ensemble models, meaning it is very good at catching unstable configurations, though at some cost to Precision. |
| **Naive Bayes** | Despite its strong (and unrealistic) assumption of feature independence and Gaussian-distributed features, it performs surprisingly well (84.45% accuracy, AUC 0.921), close to kNN. This suggests the individual features carry strong class-conditional signal even in isolation. Its Recall (0.935) is high, similar to kNN, but Precision (0.84) and MCC (0.657) are lower, indicating more false positives (predicting "unstable" when it's actually "stable"). |
| **Random Forest (Ensemble)** | Clearly the **best-performing model on every single metric** — 92.4% accuracy, AUC 0.981, F1 0.942, and by far the highest MCC (0.834), which is the most reliable metric on this moderately imbalanced dataset. By averaging many de-correlated decision trees, it captures the same non-linear interactions as a single tree but drastically reduces overfitting and produces well-calibrated probability estimates (reflected in the near-perfect AUC), while also balancing Precision and Recall much better than any other model. |
| **Overall Winner for my dataset?** | **Random Forest (Ensemble)** — it dominates all five other models across every evaluation metric (Accuracy, AUC, Precision, Recall, F1, and MCC), making it the clear and unambiguous choice for deployment on this grid-stability dataset. This is also the model used as the primary predictor in the accompanying Streamlit application. |

---

### How to Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Upload `test_data.csv` (or any similarly formatted grid measurement CSV) in the app to get predictions and a live comparison of all five models.
