"""The candidate classifiers we compare (six algorithm families).

Every model receives the same 13 features and outputs P(candidate is true).
Scaling (StandardScaler) is put inside a Pipeline for models that depend on
feature scale, so it is fitted only on the training fold (no leakage).
No class weights are used: we want honest probabilities for the HMM, and the
positive class (~10%) is not extremely rare.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED


class EpochMLP(BaseEstimator, ClassifierMixin):
    """Small neural network trained epoch by epoch with partial_fit, so we can
    record the training (and optional validation) log loss after every epoch."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, learning_rate=1e-3,
                 epochs=20, batch_size=512, random_state=RANDOM_SEED):
        self.hidden = hidden
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.array([0, 1])
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        Xv = self.scaler_.transform(X_val) if X_val is not None else None
        self.net_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                                  learning_rate_init=self.learning_rate,
                                  batch_size=self.batch_size, random_state=self.random_state)
        self.history_ = {"train_loss": [], "val_loss": []}
        for _ in range(self.epochs):
            order = rng.permutation(len(Xs))            # reshuffle every epoch
            self.net_.partial_fit(Xs[order], y[order], classes=self.classes_)
            self.history_["train_loss"].append(log_loss(y, self.net_.predict_proba(Xs)[:, 1], labels=[0, 1]))
            if Xv is not None:
                self.history_["val_loss"].append(log_loss(y_val, self.net_.predict_proba(Xv)[:, 1], labels=[0, 1]))
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models(seed=RANDOM_SEED):
    """name -> dict(family, config text, estimator, max_train_rows)."""
    return {
        "LogisticRegression": dict(
            family="Linear",
            config="StandardScaler + L2, C=1.0, lbfgs",
            model=Pipeline([("scale", StandardScaler()),
                            ("clf", LogisticRegression(C=1.0, max_iter=1000))])),
        "GaussianNB": dict(
            family="Probabilistic",
            config="Gaussian likelihoods, var_smoothing=1e-9",
            model=GaussianNB()),
        "KNN": dict(
            family="Instance-based",
            config="StandardScaler, k=25, distance-weighted, 50k train rows",
            model=Pipeline([("scale", StandardScaler()),
                            ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))]),
            # prediction cost grows with the stored training set, so KNN keeps fewer rows
            max_train_rows=50_000),
        "RandomForest": dict(
            family="Bagging ensemble",
            config="100 trees, max_depth=16, min_samples_leaf=5",
            model=RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                         n_jobs=-1, random_state=seed)),
        "LightGBM": dict(
            family="Gradient boosting",
            config="300 trees, lr=0.05, num_leaves=31",
            model=LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 n_jobs=-1, random_state=seed, verbose=-1)),
        "MLP": dict(
            family="Neural network",
            config="StandardScaler, layers 64-32 ReLU, Adam lr=1e-3, alpha=1e-4, 20 epochs",
            model=EpochMLP(hidden=(64, 32), epochs=20, random_state=seed)),
    }
