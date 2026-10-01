"""Run the pipeline on both datasets and write every table, figure and model under --out.

Usage: python run_all.py [--cic DIR] [--netrack DIR] [--out DIR] [--window N]
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from evaluate import confusion_figure, metrics, outcomes, save_table
from features import FINAL
from autoencoder import Autoencoder
from cnn import CNN
from forest import Forest
from iforest import IsoForest
from ingest import load
from preprocess import prepare, unscale
from common import WINDOW, set_seed

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cic", default="../../data/processed/cic-bell-dns-exf-2021/names",
                    help="Dataset A: the query-name CSVs written by extract_names.py")
parser.add_argument("--netrack", default="../../data/netrack-learn", help="Dataset B: the netrack/learn repository")
parser.add_argument("--out", default="../output", help="where tables/, figures/ and models/ are written")
parser.add_argument("--window", type=int, default=WINDOW, help="rows looked back by same_sld_count")
args = parser.parse_args()

out = Path(args.out)
tables, figures, models = out / "tables", out / "figures", out / "models"
for folder in (tables, figures, models):
    folder.mkdir(parents=True, exist_ok=True)

set_seed()
# Tuned settings from tune.py, if it has been run; otherwise every model uses its PARAMS.
best_file = models / "best_params.json"
best = json.loads(best_file.read_text()) if best_file.exists() else {}
# Dataset name, where it is, and whether it has the query type (only B does).
DATASETS = [("A", args.cic, False), ("B", args.netrack, True)]
MODELS = [Forest, CNN, IsoForest, Autoencoder]

results, thresholds = [], {}
for dataset, path, with_qtype in DATASETS:
    train, test = prepare(load(dataset, path, args.window), with_qtype, args.window)
    # Every test row in the original units, with each model's outcome added as a column below.
    verdicts = unscale(test, args.window)[["qname", "subdomain", "capture", "label", "truncated", *FINAL, "qname_len", "qtype"]]
    for Model in MODELS:
        started = time.time()
        model = Model(with_qtype, **best.get(f"{Model.name}_{dataset}", {})).fit(train)
        print(f"{model.name}, dataset {dataset}: trained in {time.time() - started:.0f} s")
        thresholds[f"{model.name}_{dataset}"] = model.threshold
        model.save(models / f"{model.name}_{dataset}.joblib")
        score = model.score(test)
        results.append({"model": model.name, "dataset": dataset, **metrics(test["label"], score, model.threshold)})
        verdicts[model.name] = outcomes(test["label"], score, model.threshold)
    # The rows at least one model got wrong, with every model's outcome side by side.
    names = [Model.name for Model in MODELS]
    wrong = verdicts[names].isin(["FP", "FN"]).any(axis=1)
    verdicts[wrong].to_csv(tables / f"errors_{dataset}.csv", index=False)

save_table(results, tables / "performance.csv")
confusion_figure(results, figures / "confusion_matrices.png")
(models / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

for table in sorted(tables.glob("*.csv")):
    print(f"\n== {table.stem}\n{pd.read_csv(table).to_string(index=False)}")
