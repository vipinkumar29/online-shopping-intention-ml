"""Decision Tree model.

max_depth and min_samples_leaf keep the tree from growing until it memorises the
training data. class_weight="balanced" handles the class imbalance.

    python model/decision_tree.py
"""

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, build_preprocessor, run

MODEL_NAME = "Decision Tree"
MODEL_SLUG = "decision_tree"


def build_model():
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


if __name__ == "__main__":
    run(MODEL_NAME, MODEL_SLUG, build_model())
