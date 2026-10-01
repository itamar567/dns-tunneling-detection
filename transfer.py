"""Cross-dataset test: train every model on one dataset and test it on the other.

Only the six shared features are used (the query type exists in B only), and scaling is by
fixed protocol limits, so nothing is refitted on the target dataset. Each model uses the
tuned settings of the dataset it is trained on. Writes tables/transfer.csv under --out.

Usage: python transfer.py [--cic DIR] [--netrack DIR] [--out DIR] [--window N]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from autoencoder import Autoencoder
from cnn import CNN
from common import WINDOW, set_seed
from evaluate import metrics, save_table
from forest import Forest
from iforest import IsoForest
from ingest import load
from preprocess import prepare

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cic", default="../../data/processed/cic-bell-dns-exf-2021/names")
parser.add_argument("--netrack", default="../../data/netrack-learn")
parser.add_argument("--out", default="../output")
parser.add_argument("--window", type=int, default=WINDOW)
args = parser.parse_args()

out = Path(args.out)
(out / "tables").mkdir(parents=True, exist_ok=True)
best_file = out / "models" / "best_params.json"
best = json.loads(best_file.read_text()) if best_file.exists() else {}

set_seed()
MODELS = [Forest, CNN, IsoForest, Autoencoder]
data = {name: prepare(load(name, path, args.window), with_qtype=False, window=args.window)
        for name, path in (("A", args.cic), ("B", args.netrack))}

rows = []
for source, target in (("A", "A"), ("B", "B"), ("A", "B"), ("B", "A")):
    train, test = data[source][0], data[target][1]
    for Model in MODELS:
        model = Model(False, **best.get(f"{Model.name}_{source}", {})).fit(train)
        result = metrics(test["label"], model.score(test), model.threshold)
        rows.append({"model": model.name, "train": source, "test": target, **result})
        print(f"{model.name}: train {source}, test {target}: F1 {result['f1']}, FPR {result['fpr']}, FNR {result['fnr']}")

save_table(rows, out / "tables" / "transfer.csv")
print(pd.DataFrame(rows).pivot(index="model", columns=["train", "test"], values="f1").to_string())
