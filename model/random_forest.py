"""Random Forest model.

An ensemble of trees on bootstrap samples with random feature subsets, which
usually beats a single tree. min_samples_leaf=5 keeps each tree (and the saved
file) smaller and reduces overfitting. class_weight="balanced" for the imbalance.

    python model/random_forest.py
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from common import RANDOM_STATE, build_preprocessor, run

MODEL_NAME = "Random Forest"
MODEL_SLUG = "random_forest"


def build_model():
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


if __name__ == "__main__":
    run(MODEL_NAME, MODEL_SLUG, build_model())
