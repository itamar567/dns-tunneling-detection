"""Autoencoder on the six final features (plus the one-hot query type when with_qtype).

It learns to compress and rebuild benign rows only. A row it rebuilds badly does not look
benign, so the rebuild error is the score. Attack labels are never used in training."""
import torch
from torch import nn

from features import feature_matrix
from common import SEED, Model, holdout, train_torch

# Threshold = mean + k std of the rebuild error on held-out benign rows
PARAMS = dict(hidden=(32, 16), bottleneck=4, epochs=50, batch_size=256, lr=1e-3, k=2.0, holdout=0.2)


class Autoencoder(Model):
    name = "autoencoder"
    grid = {"bottleneck": [2, 4, 8], "k": [1.0, 2.0, 3.0]}

    def __init__(self, with_qtype, **params):
        self.with_qtype = with_qtype
        self.params = {**PARAMS, **params}

    def build(self, width):
        """width -> hidden -> bottleneck -> hidden reversed -> width, ReLU between layers."""
        sizes = [width, *self.params["hidden"], self.params["bottleneck"], *reversed(self.params["hidden"]), width]
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        return nn.Sequential(*layers[:-1])  # no ReLU after the output layer

    def errors(self, X):
        """Mean squared rebuild error per row."""
        with torch.no_grad():
            return ((self.net(X) - X) ** 2).mean(dim=1).numpy()

    def fit(self, df):
        p = self.params
        fit_rows, check_rows = holdout(df[df["label"] == 0], p["holdout"])
        X = torch.tensor(feature_matrix(fit_rows, self.with_qtype)[0])
        torch.manual_seed(SEED)  # same starting weights every run
        self.net = self.build(X.shape[1])
        train_torch(self.net, (X,), lambda net, x: nn.functional.mse_loss(net(x), x), p["epochs"], p["batch_size"], p["lr"])
        errors = self.errors(torch.tensor(feature_matrix(check_rows, self.with_qtype)[0]))
        self.threshold = float(errors.mean() + p["k"] * errors.std())
        return self

    def score(self, df):
        X, _ = feature_matrix(df, self.with_qtype)
        return self.errors(torch.tensor(X))
