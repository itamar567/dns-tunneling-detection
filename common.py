import os
import random
import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

SEED = 1
WINDOW = 200  # rows looked back by same_sld_count (chosen by optimize.py from 50, 100, 200)


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch, and make PyTorch deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def holdout(df, share):
    """Split off `share` of the rows, to set a threshold on rows the model was not fitted on.
    With share 0 both parts are the whole table."""
    if share == 0:
        return df, df
    return train_test_split(df, test_size=share, random_state=SEED)


class Model:
    """What every model shares: a decision threshold on its score, and saving to one file."""
    threshold = 0.5

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def train_torch(net, inputs, loss_fn, epochs, batch_size, lr):
    """Train a PyTorch network with Adam on mini-batches, in a shuffled order that repeats from run to run.
    `inputs` is a tuple of tensors with one row per sample; loss_fn(net, *batch) returns the batch loss."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    order = torch.Generator().manual_seed(SEED)
    net.train()
    for _ in range(epochs):
        for rows in torch.randperm(len(inputs[0]), generator=order).split(batch_size):
            optimizer.zero_grad()
            loss_fn(net, *(t[rows] for t in inputs)).backward()
            optimizer.step()
    net.eval()
    return net
