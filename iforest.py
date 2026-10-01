"""Isolation Forest on the six final features (plus the one-hot query type when with_qtype).

It learns what benign rows look like and scores each row by how easy it is to isolate.
Attack labels are never used in training."""
import numpy as np
from sklearn.ensemble import IsolationForest

from features import feature_matrix
from common import SEED, Model, holdout

PARAMS = dict(n_estimators=200, max_samples=256, max_features=1.0, bootstrap=False, random_state=SEED, n_jobs=1)
TARGET_FPR = 0.02  # share of benign rows allowed above the threshold
HOLDOUT = 0.2     # share of the benign training rows kept aside to set the threshold on


class IsoForest(Model):
    name = "isolation_forest"
    grid = {"target_fpr": [0.01, 0.02, 0.05, 0.10], "n_estimators": [50, 100, 200, 500]}

    def __init__(self, with_qtype, target_fpr=TARGET_FPR, holdout=HOLDOUT, **params):
        self.with_qtype = with_qtype
        self.target_fpr = target_fpr
        self.holdout = holdout
        self.model = IsolationForest(**{**PARAMS, **params})

    def fit(self, df):
        fit_rows, check_rows = holdout(df[df["label"] == 0], self.holdout)
        self.model.fit(feature_matrix(fit_rows, self.with_qtype)[0])
        check_scores = self.score_matrix(feature_matrix(check_rows, self.with_qtype)[0])
        self.threshold = float(np.quantile(check_scores, 1 - self.target_fpr))
        return self

    def score_matrix(self, X):
        """Anomaly score: higher means easier to isolate, so less like the benign rows."""
        return -self.model.score_samples(X)

    def score(self, df):
        X, _ = feature_matrix(df, self.with_qtype)
        return self.score_matrix(X)
