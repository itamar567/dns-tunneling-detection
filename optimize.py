"""Optimization steps from the error analysis, applied one after another.

Starts from the pipeline as it was before any fix and switches the fixes on in order. After
each step every model is trained and tested on both datasets. The window for step 3 is
chosen by 5-fold cross-validation on the training part, not on the test part.

  0  baseline: labels by capture file, one-label suffixes, window 100, thresholds on training rows
  1  label Dataset A by content (background traffic inside attack captures is benign)
  2  two-part country suffixes (co.uk, com.br) count as one suffix
  3  window chosen from 50, 100, 200
  4  unsupervised thresholds set on held-out benign rows

Writes tables/optimization_steps.csv and tables/window_choice.csv under --out.

Usage: python optimize.py [--cic DIR] [--netrack DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import features
import ingest
from autoencoder import Autoencoder
from cnn import CNN
from common import set_seed
from evaluate import metrics, save_table
from forest import Forest
from iforest import IsoForest
from ingest import load
from preprocess import folds, prepare

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cic", default="../../data/processed/cic-bell-dns-exf-2021/names")
parser.add_argument("--netrack", default="../../data/netrack-learn")
parser.add_argument("--out", default="../output")
args = parser.parse_args()

out = Path(args.out)
(out / "tables").mkdir(parents=True, exist_ok=True)
best_file = out / "models" / "best_params.json"
best = json.loads(best_file.read_text()) if best_file.exists() else {}

set_seed()
DATASETS = [("A", args.cic, False), ("B", args.netrack, True)]
MODELS = [Forest, CNN, IsoForest, Autoencoder]
FEATURE_MODELS = [Forest, IsoForest, Autoencoder]  # the CNN reads the name, so the window does not reach it
COUNTRY_SECOND_LEVEL = features.COUNTRY_SECOND_LEVEL


def settings(content_labels, two_part_suffixes):
    ingest.LABEL_BY_CONTENT = content_labels
    features.COUNTRY_SECOND_LEVEL = COUNTRY_SECOND_LEVEL if two_part_suffixes else set()


def build(Model, dataset, with_qtype, holdout):
    params = best.get(f"{Model.name}_{dataset}", {})
    if Model in (IsoForest, Autoencoder):
        params = {**params, "holdout": holdout}
    return Model(with_qtype, **params)


def evaluate_step(step, window, holdout, rows):
    for dataset, path, with_qtype in DATASETS:
        train, test = prepare(load(dataset, path, window), with_qtype, window)
        for Model in MODELS:
            model = build(Model, dataset, with_qtype, holdout).fit(train)
            result = metrics(test["label"], model.score(test), model.threshold)
            rows.append({"step": step, "model": model.name, "dataset": dataset,
                         **{k: result[k] for k in ("f1", "fpr", "fnr", "FP", "FN")}})
            print(f"step {step}, {model.name}, {dataset}: F1 {result['f1']}, FPR {result['fpr']}, FNR {result['fnr']}")


def choose_window(candidates=(50, 100, 200)):
    """Mean 5-fold validation F1 of the feature models on both training parts, per window."""
    choice = []
    for window in candidates:
        f1 = []
        for dataset, path, with_qtype in DATASETS:
            train, _ = prepare(load(dataset, path, window), with_qtype, window)
            for Model in FEATURE_MODELS:
                for fit_rows, check_rows in folds(train):
                    fit, check = train.iloc[fit_rows], train.iloc[check_rows]
                    model = build(Model, dataset, with_qtype, holdout=0).fit(fit)
                    f1.append(metrics(check["label"], model.score(check), model.threshold)["f1"])
        choice.append({"window": window, "f1_mean": round(float(np.mean(f1)), 4)})
        print(f"window {window}: mean validation F1 {choice[-1]['f1_mean']}")
    return max(choice, key=lambda c: c["f1_mean"])["window"], choice


rows = []
settings(content_labels=False, two_part_suffixes=False)
evaluate_step("0 baseline", window=100, holdout=0, rows=rows)
settings(content_labels=True, two_part_suffixes=False)
evaluate_step("1 label A by content", window=100, holdout=0, rows=rows)
settings(content_labels=True, two_part_suffixes=True)
evaluate_step("2 two-part suffixes", window=100, holdout=0, rows=rows)
window, choice = choose_window()
save_table(choice, out / "tables" / "window_choice.csv")
evaluate_step(f"3 window {window}", window=window, holdout=0, rows=rows)
evaluate_step("4 held-out thresholds", window=window, holdout=0.2, rows=rows)
save_table(rows, out / "tables" / "optimization_steps.csv")
print(f"chosen window: {window}")
