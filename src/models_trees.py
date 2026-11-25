"""Tree models: decision tree, random forest, gradient boosting.

Owner: Member 3 (tree models).
"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocess import tree_steps


def decision_tree():
    return Pipeline(tree_steps() + [("clf", DecisionTreeClassifier(random_state=C.SEED))])


def random_forest():
    return Pipeline(tree_steps() + [("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=C.SEED))])


def gradient_boosting():
    # Handles missing values natively, so no imputer.
    return Pipeline([("clf", HistGradientBoostingClassifier(max_iter=400, early_stopping=False, random_state=C.SEED))])


MODELS = {
    "Decision tree": (
        decision_tree,
        {"clf__max_depth": [3, 5, 8, 12], "clf__min_samples_leaf": [1, 10, 30], "clf__class_weight": [None, "balanced"]},
    ),
    "Random forest": (
        random_forest,
        {"clf__max_features": ["sqrt", 0.5], "clf__min_samples_leaf": [1, 3, 10], "clf__class_weight": [None, "balanced_subsample"]},
    ),
    "Gradient boosting": (
        gradient_boosting,
        {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [15, 31], "clf__l2_regularization": [0.0, 1.0]},
    ),
}
