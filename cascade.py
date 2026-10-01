"""The multi-stage cascade.

Stage 1  Isolation Forest, a cheap filter with a wide threshold: rows it finds normal are benign.
Stage 2  Random Forest and CNN look at the rest. When they agree and the Random Forest is sure,
         that is the answer.
Stage 3  Rows they disagree on, or where the Random Forest probability is inside BAND, are
         escalated. Until the LLM arbiter exists the Random Forest's vote decides them.

Runs within each dataset and across datasets (train on one, test on the other), with the six
shared features only, so all four cases use the same models. Writes tables/cascade.csv (the
single models against the cascade), tables/cascade_stages.csv (rows reaching and settled at each
stage) and tables/escalated_<train>_<test>.csv (the escalated rows, the LLM's input) under --out.

Usage: python cascade.py [--cic DIR] [--netrack DIR] [--out DIR] [--window N]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cnn import CNN
from common import WINDOW, set_seed
from evaluate import metrics, save_table
from features import FINAL
from forest import Forest
from iforest import IsoForest
from ingest import load
from preprocess import prepare, unscale

STAGE1_FPR = 0.10     # stage 1 passes on 10% of the benign training rows, so almost no tunnel is filtered out
BAND = (0.35, 0.65)   # Random Forest probabilities treated as unsure

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cic", default="../../data/processed/cic-bell-dns-exf-2021/names")
parser.add_argument("--netrack", default="../../data/netrack-learn")
parser.add_argument("--out", default="../output")
parser.add_argument("--window", type=int, default=WINDOW)
args = parser.parse_args()


def cascade(iforest, forest, cnn, df):
    """Run the three stages. Returns the decision per row (1 = tunnel), the stage that decided it
    (1, 2 or 3) and the three models' scores."""
    scores = {"isolation_forest": iforest.score(df), "random_forest": forest.score(df), "cnn": cnn.score(df)}
    suspicious = scores["isolation_forest"] >= iforest.threshold
    forest_vote = scores["random_forest"] >= forest.threshold
    cnn_vote = scores["cnn"] >= cnn.threshold
    unsure = (scores["random_forest"] > BAND[0]) & (scores["random_forest"] < BAND[1])
    escalate = suspicious & ((forest_vote != cnn_vote) | unsure)
    stage = np.where(~suspicious, 1, np.where(escalate, 3, 2))
    decision = np.where(stage == 1, 0, forest_vote).astype(int)  # stage 3: the Random Forest's vote for now
    return decision, stage, scores


out = Path(args.out)
(out / "tables").mkdir(parents=True, exist_ok=True)
best_file = out / "models" / "best_params.json"
best = json.loads(best_file.read_text()) if best_file.exists() else {}

set_seed()
data = {name: prepare(load(name, path, args.window), with_qtype=False, window=args.window)
        for name, path in (("A", args.cic), ("B", args.netrack))}

results, stages = [], []
for source, target in (("A", "A"), ("B", "B"), ("A", "B"), ("B", "A")):
    train, test = data[source][0], data[target][1]
    params = lambda name: best.get(f"{name}_{source}", {})
    iforest = IsoForest(False, **{**params("isolation_forest"), "target_fpr": STAGE1_FPR}).fit(train)
    forest = Forest(False, **params("random_forest")).fit(train)
    cnn = CNN(False, **params("cnn")).fit(train)
    decision, stage, scores = cascade(iforest, forest, cnn, test)

    case = {"train": source, "test": target}
    for name, model in (("random_forest", forest), ("cnn", cnn)):
        results.append({**case, "configuration": name, **metrics(test["label"], scores[name], model.threshold), "escalated": 0})
    results.append({**case, "configuration": "cascade (stage 3 = random_forest)",
                    **metrics(test["label"], decision, 0.5), "escalated": int((stage == 3).sum())})
    for s in (1, 2, 3):
        stages.append({**case, "stage": s, "rows_reaching": int((stage >= s).sum()), "rows_settled": int((stage == s).sum())})

    escalated = unscale(test[stage == 3], args.window)[["qname", "subdomain", "capture", "label", *FINAL]]
    for name, score in scores.items():
        escalated[f"{name}_score"] = score[stage == 3]
    escalated.to_csv(out / "tables" / f"escalated_{source}_{target}.csv", index=False)
    print(f"train {source}, test {target}: escalated {len(escalated)}; "
          f"cascade F1 {results[-1]['f1']} FPR {results[-1]['fpr']}; forest F1 {results[-3]['f1']} FPR {results[-3]['fpr']}")

save_table(results, out / "tables" / "cascade.csv")
save_table(stages, out / "tables" / "cascade_stages.csv")
