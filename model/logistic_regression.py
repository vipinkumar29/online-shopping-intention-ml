"""Logistic Regression model.

A linear baseline. class_weight="balanced" is used because only ~15% of the
sessions end in a purchase, so we don't want the model to ignore the minority
class. Features are scaled inside the shared preprocessing pipeline.

    python model/logistic_regression.py
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import RANDOM_STATE, build_preprocessor, run

MODEL_NAME = "Logistic Regression"
MODEL_SLUG = "logistic_regression"


def build_model():
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


if __name__ == "__main__":
    run(MODEL_NAME, MODEL_SLUG, build_model())
