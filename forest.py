"""Random Forest on the six final features (plus the one-hot query type when with_qtype)."""
from sklearn.ensemble import RandomForestClassifier

from features import feature_matrix
from common import SEED, Model

# n_jobs=1 so the results repeat exactly from run to run.
PARAMS = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=5, max_depth=None,
              class_weight="balanced_subsample", random_state=SEED, n_jobs=1)


class Forest(Model):
    name = "random_forest"
    grid = {"max_depth": [3, 5, 10, 20, None], "n_estimators": [50, 100, 300, 500]}

    def __init__(self, with_qtype, threshold=0.5, **params):
        self.with_qtype = with_qtype
        self.threshold = threshold
        self.model = RandomForestClassifier(**{**PARAMS, **params})

    def fit(self, df):
        X, y = feature_matrix(df, self.with_qtype)
        self.model.fit(X, y)
        return self

    def score(self, df):
        """Probability that each row is a tunnel."""
        X, _ = feature_matrix(df, self.with_qtype)
        return self.model.predict_proba(X)[:, 1]
