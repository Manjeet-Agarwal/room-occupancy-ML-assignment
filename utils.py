"""Helpers shared by notebooks/train_models.ipynb and app.py.

Keeping the column lists and the metric code in one place guarantees that the
notebook and the Streamlit app score a model in exactly the same way.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "Room_Occupancy_Count"

FEATURE_COLUMNS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope",
    "S6_PIR", "S7_PIR",
]

# display name -> file name inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def score_predictions(y_true, y_pred, y_proba, classes):
    """Six metrics from labels and class probabilities.

    Precision, Recall and F1 use macro averaging (every class counts equally).
    AUC is the macro average of one-vs-rest ROC AUCs. If a class is missing
    from y_true, its ROC curve is undefined, so AUC is returned as NaN.
    """
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                            average="macro", labels=classes)
    except ValueError:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X, y):
    """Predict with a fitted model and return the six metrics as a dict."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return score_predictions(y, y_pred, y_proba, model.classes_)
