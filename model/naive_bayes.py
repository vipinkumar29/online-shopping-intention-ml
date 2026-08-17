"""Gaussian Naive Bayes model.

GaussianNB is the natural choice because the numeric features are continuous. It
is fast to train and gives us a probabilistic baseline to compare against.

    python model/naive_bayes.py
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from common import build_preprocessor, run

MODEL_NAME = "Naive Bayes"
MODEL_SLUG = "naive_bayes"


def build_model():
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("classifier", GaussianNB()),
        ]
    )


if __name__ == "__main__":
    run(MODEL_NAME, MODEL_SLUG, build_model())
