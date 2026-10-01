"""Hyperparameter tuning and sensitivity: 5-fold cross-validation on the training part only.

Each parameter in a model's `grid` is varied on its own while the others keep their PARAMS
value. For every value we record the mean and standard deviation of F1 across the 5 folds;
the value with the best mean wins. Writes tables/sensitivity.csv and models/best_params.json
under --out. run_all.py then trains with the winners.

Usage: python tune.py [--cic DIR] [--netrack DIR] [--out DIR] [--window N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder import Autoencoder
from cnn import CNN
from common import WINDOW, set_seed
from evaluate import metrics
from forest import Forest
from iforest import IsoForest
from ingest import load
from preprocess import folds, prepare

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cic", default="../../data/processed/cic-bell-dns-exf-2021/names")
parser.add_argument("--netrack", default="../../data/netrack-learn")
parser.add_argument("--out", default="../output")
parser.add_argument("--window", type=int, default=WINDOW)
args = parser.parse_args()

out = Path(args.out)
for folder in ("tables", "models"):
    (out / folder).mkdir(parents=True, exist_ok=True)

set_seed()
DATASETS = [("A", args.cic, False), ("B", args.netrack, True)]
MODELS = [Forest, CNN, IsoForest, Autoencoder]


def cross_validate(Model, with_qtype, train, params):
    """F1 on each validation fold for a model built with `params`."""
    scores = []
    for fit_rows, check_rows in folds(train):
        fit, check = train.iloc[fit_rows], train.iloc[check_rows]
        model = Model(with_qtype, **params).fit(fit)
        scores.append(metrics(check["label"], model.score(check), model.threshold)["f1"])
    return scores


def tune():
    """Run the grids; write the sensitivity table and the best value of every parameter."""
    rows, best = [], {}
    for dataset, path, with_qtype in DATASETS:
        train, _ = prepare(load(dataset, path, args.window), with_qtype, args.window)  # the test part is never touched here
        for Model in MODELS:
            best[f"{Model.name}_{dataset}"] = {}
            for param, values in Model.grid.items():
                results = []
                for value in values:
                    started = time.time()
                    f1 = cross_validate(Model, with_qtype, train, {param: value})
                    results.append((value, np.mean(f1), np.std(f1)))
                    rows.append({"model": Model.name, "dataset": dataset, "param": param, "value": value,
                                 "f1_mean": round(np.mean(f1), 4), "f1_std": round(np.std(f1), 4)})
                    print(f"{Model.name} {dataset} {param}={value}: F1 {np.mean(f1):.4f} with STD {np.std(f1):.4f} ({time.time() - started:.0f} s)")
                best[f"{Model.name}_{dataset}"][param] = max(results, key=lambda r: r[1])[0]

    pd.DataFrame(rows).to_csv(out / "tables" / "sensitivity.csv", index=False)
    (out / "models" / "best_params.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


tune()
