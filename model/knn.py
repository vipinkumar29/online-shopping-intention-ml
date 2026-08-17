"""k-Nearest Neighbors model.

Scaling matters a lot here since kNN works on distances (handled by the shared
preprocessing). k is set to roughly sqrt(number of training rows) and forced to
an odd number so votes can't tie.

    python model/knn.py
"""

from math import sqrt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from common import (
    build_preprocessor,
    evaluate,
    export_test_data,
    load_split,
    save_model,
    update_metrics,
)

MODEL_NAME = "kNN"
MODEL_SLUG = "knn"


def build_model(k):
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("classifier", KNeighborsClassifier(n_neighbors=k)),
        ]
    )


def main():
    X_train, X_test, y_train, y_test = load_split()
    export_test_data(X_test, y_test)

    k = int(sqrt(len(X_train)))
    if k % 2 == 0:
        k += 1

    model = build_model(k)
    model.fit(X_train, y_train)
    scores = evaluate(model, X_test, y_test)

    save_model(model, MODEL_SLUG)
    update_metrics(MODEL_NAME, scores)

    print(f"{MODEL_NAME} (k={k}):")
    print("  " + "  ".join(f"{key}={val}" for key, val in scores.items()))
    print(f"  saved -> model/{MODEL_SLUG}.pkl")


if __name__ == "__main__":
    main()
