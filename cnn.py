"""The 1D-CNN from Paper B. It reads the query name itself instead of the six features."""
import numpy as np
import torch
from torch import nn

from preprocess import class_weights
from common import SEED, Model, train_torch

# Each character becomes its index in ALPHABET; 0 is padding.
NAME_LEN = 40
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.+/="
VOCAB_SIZE = len(ALPHABET) + 2

# Paper B's settings
PARAMS = dict(embedding=100, filters=1024, kernel=4, dense=256, epochs=10, batch_size=128, lr=1e-3, threshold=0.9)


def encode_names(subdomains):
    """Character indices of the first NAME_LEN characters of each subdomain, padded with 0."""
    index = {ch: i + 1 for i, ch in enumerate(ALPHABET)}
    out = np.zeros((len(subdomains), NAME_LEN), dtype=np.int64)
    for row, text in enumerate(subdomains.fillna("")):
        for col, ch in enumerate(str(text)[:NAME_LEN]):
            out[row, col] = index.get(ch, VOCAB_SIZE - 1)
    return out


class Network(nn.Module):
    """Embedding -> Conv1D (ReLU) -> Flatten -> Dense (ReLU) -> Dense, as in Paper B. The last layer
    returns a logit; the sigmoid is applied in the loss and in CNN.score."""

    def __init__(self, embedding, filters, kernel, dense):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embedding, padding_idx=0)
        self.conv = nn.Conv1d(embedding, filters, kernel)
        self.dense = nn.Linear(filters * (NAME_LEN - kernel + 1), dense)
        self.out = nn.Linear(dense, 1)

    def forward(self, names):
        x = self.embed(names).transpose(1, 2)            # (rows, embedding, characters)
        x = torch.relu(self.conv(x)).flatten(1)
        return self.out(torch.relu(self.dense(x))).squeeze(1)


class CNN(Model):
    name = "cnn"
    grid = {"filters": [128, 256, 512]}

    def __init__(self, with_qtype=None, **params):  # the CNN ignores the query type
        self.params = {**PARAMS, **params}
        self.threshold = self.params["threshold"]

    def fit(self, df):
        p = self.params
        names, y = torch.tensor(encode_names(df["subdomain"])), torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
        weights = class_weights(df["label"])
        row_weight = torch.where(y == 1, weights[1], weights[0]).float()

        def loss(net, x, target, weight):
            return nn.functional.binary_cross_entropy_with_logits(net(x), target, weight=weight)

        torch.manual_seed(SEED)  # same starting weights every run
        self.net = Network(p["embedding"], p["filters"], p["kernel"], p["dense"])
        train_torch(self.net, (names, y, row_weight), loss, p["epochs"], p["batch_size"], p["lr"])
        return self

    def score(self, df):
        """Probability that each row is a tunnel."""
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.tensor(encode_names(df["subdomain"])))).numpy()
