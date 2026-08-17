"""Streamlit app for the Online Shoppers Purchasing Intention project.

It loads the models trained by the scripts in model/ and lets you pick a model,
upload a test CSV, and see the metrics, confusion matrix and classification
report on that data.

    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data.csv"
TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(
    page_title="Online Shoppers Purchasing Intention",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_reference_metrics():
    path = MODEL_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("🛒 Online Shoppers Purchasing Intention")
    st.caption(
        "Predict whether a browsing session ends in a purchase (Revenue). "
        "Upload test data, pick a model, and check how it performs."
    )

    models = load_models()
    if not models:
        st.error(
            "No trained models found in model/. Run the scripts in model/ first, "
            "e.g. `python model/random_forest.py`."
        )
        st.stop()

    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox("Select a model", list(models.keys()))

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
    st.sidebar.caption(
        "The CSV needs the original feature columns and a Revenue column. "
        "Leave it empty to use the bundled test_data.csv."
    )

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        source = "uploaded file"
    elif DEFAULT_TEST_DATA.exists():
        data = pd.read_csv(DEFAULT_TEST_DATA)
        source = "bundled test_data.csv"
    else:
        st.info("Upload a CSV file to begin.")
        st.stop()

    st.subheader("Input data")
    st.write(f"Using **{source}** — {data.shape[0]} rows, {data.shape[1]} columns.")
    st.dataframe(data.head(), use_container_width=True)

    if TARGET not in data.columns:
        st.error(f"The uploaded data needs a `{TARGET}` column to score the model.")
        st.stop()

    X = data.drop(columns=[TARGET])
    y_true = data[TARGET].astype(int)

    model = models[model_name]
    try:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
    except Exception as exc:
        st.error(f"Could not run the model on this data: {exc}")
        st.stop()

    st.subheader(f"Evaluation metrics — {model_name}")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, f"{value:.4f}")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No purchase", "Purchase"],
            yticklabels=["No purchase", "Purchase"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("#### Classification report")
        report = classification_report(
            y_true,
            y_pred,
            target_names=["No purchase", "Purchase"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    reference = load_reference_metrics()
    if reference:
        st.subheader("All models on the held-out test set")
        st.caption("Scores saved during training (model/metrics.json).")
        ref_df = pd.DataFrame(reference).transpose()
        st.dataframe(
            ref_df.style.highlight_max(axis=0, color="#c6f6d5"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
