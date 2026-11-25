"""Classical models: majority baseline, logistic regression, k-nearest neighbours.

Owner: Member 2 (classical models).
Each entry gives a pipeline factory and the hyper-parameter grid searched with
star-grouped CV on the development set (see experiments/e2_models.py).
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from . import config as C
from .preprocess import scaled_steps, tree_steps


def baseline():
    # 'prior' predicts the majority class and outputs the class frequency as its probability.
    return Pipeline(tree_steps() + [("clf", DummyClassifier(strategy="prior"))])


def logistic():
    return Pipeline(scaled_steps() + [("clf", LogisticRegression(max_iter=5000, random_state=C.SEED))])


def knn():
    return Pipeline(scaled_steps() + [("clf", KNeighborsClassifier())])


MODELS = {
    "Majority baseline": (baseline, {}),
    "Logistic regression": (
        logistic,
        {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    ),
    "k-NN": (
        knn,
        {"clf__n_neighbors": [5, 15, 31, 51, 101], "clf__weights": ["uniform", "distance"]},
    ),
}
