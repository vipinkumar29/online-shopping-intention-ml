"""Helpers shared by the individual model scripts.

Every model lives in its own file (logistic_regression.py, decision_tree.py,
knn.py, naive_bayes.py, random_forest.py) and imports the helpers here so they
all use the same feature lists, preprocessing, train/test split and metrics.

Usage (from the project root):
    python model/logistic_regression.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Continuous columns -> scaled. Kept separate because kNN and Logistic
# Regression are sensitive to feature scale.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# The integer-coded columns (OperatingSystems, Browser, ...) are category ids,
# not ordinal values, so they are one-hot encoded like the text columns.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# Order used for the comparison table.
MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
]


def build_preprocessor():
    """Scale the numeric columns and one-hot encode the categorical ones.

    sparse_output=False so the matrix stays dense, which GaussianNB needs.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_split():
    """Load the dataset and return the same 80/20 stratified split every time."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def evaluate(model, X_test, y_test):
    """Return the six metrics required by the assignment."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_test, y_proba)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }


def save_model(model, slug):
    """Save a fitted pipeline to model/<slug>.pkl."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{slug}.pkl"
    joblib.dump(model, path)
    return path


def export_test_data(X_test, y_test):
    """Write the test split to test_data.csv (this is what the app loads)."""
    test_df = X_test.copy()
    test_df[TARGET] = y_test.astype(bool)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    return TEST_DATA_PATH


def update_metrics(name, scores):
    """Add or replace one model's scores in metrics.json, keeping the order."""
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    metrics[name] = scores

    ordered = {m: metrics[m] for m in MODEL_ORDER if m in metrics}
    ordered.update({k: v for k, v in metrics.items() if k not in ordered})

    with open(METRICS_PATH, "w") as f:
        json.dump(ordered, f, indent=2)


def run(model_name, model_slug, model):
    """Fit a model on the training split, score it and save the artifacts."""
    X_train, X_test, y_train, y_test = load_split()
    export_test_data(X_test, y_test)

    model.fit(X_train, y_train)
    scores = evaluate(model, X_test, y_test)

    save_model(model, model_slug)
    update_metrics(model_name, scores)

    print(f"{model_name}:")
    print("  " + "  ".join(f"{k}={v}" for k, v in scores.items()))
    print(f"  saved -> model/{model_slug}.pkl")
    return scores
