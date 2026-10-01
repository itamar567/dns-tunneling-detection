"""Stage 3 of the cascade: a local LLM decides the rows the models could not settle.

Reads tables/escalated_<train>_<test>.csv (written by cascade.py) and tables/cascade.csv, and writes
under --out: tables/llm_arbiter.csv, tables/llm_cascade.csv, tables/llm_prompt_sensitivity.csv.

Usage: python arbiter.py [--out DIR] [--workers N] [--sample N] [--max-rows N]
"""
import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import SEED, WINDOW

URL = "http://localhost:12434/engines/v1/chat/completions"
MODEL = "ai/qwen3.6:35b-a3b"
OPTIONS = dict(temperature=0.0, top_p=0.9, max_tokens=512, seed=SEED)
CASES = [("A", "A"), ("B", "B"), ("B", "A"), ("A", "B")]

SYSTEM = """You are a DNS security analyst. You decide whether one DNS query is part of a DNS tunnel
(data or commands hidden in the query name) or ordinary traffic.
Protocol facts: a DNS name has at most 253 characters and each label at most 63. Ordinary names
have a few labels made of words, brands or short codes. Encoded payloads produce long labels of
random-looking characters, many digits, mixed case and high entropy. A tunnel client sends many
different names under one domain in a short time.
Think through the evidence in a few short points, then decide. Reply with one JSON object only:
{"reasons": ["<short point>", "<short point>", "<short point>"], "verdict": "tunnel" or "benign", "confidence": <0.0 to 1.0>}"""

VARIANTS = ["full", "no_votes", "no_name", "no_window", "reordered"]


def user_prompt(row, variant="full"):
    """The telemetry for one row. The label, the capture name and the domain are never included."""
    name = f"Query name (subdomain part): {row.subdomain}"
    features = [f"subdomain length = {row.subdomain_len:.0f} characters (limit 253)",
                f"name entropy = {row.qname_entropy:.2f} bits per character (about 6 at most)",
                f"labels = {row.label_count:.0f}",
                f"longest label = {row.label_max:.0f} characters (limit 63)",
                f"average label length = {row.label_avg:.1f} characters"]
    window = f"queries to the same domain among the last {WINDOW} = {row.same_sld_count:.0f}"
    votes = (f"Model scores: Random Forest p(tunnel) = {row.random_forest_score:.2f}, "
             f"CNN p(tunnel) = {row.cnn_score:.2f}, Isolation Forest anomaly score = {row.isolation_forest_score:.2f}")
    if variant != "no_window":
        features.append(window)
    if variant == "reordered":
        features = features[::-1]
    parts = ([] if variant == "no_name" else [name]) + ["Features: " + "; ".join(features) + "."]
    if variant != "no_votes":
        parts.append(votes + ". The models disagree or are unsure about this query.")
    return "\n".join(parts)


class Arbiter:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry["reply"]

    def ask(self, system, user):
        """The model's reply text and the seconds it took (0 when it came from the cache)."""
        body = {"model": MODEL, **OPTIONS, "messages": [{"role": "system", "content": system},
                                                        {"role": "user", "content": user}],
                "chat_template_kwargs": {"enable_thinking": False}}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key], 0.0
        request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        started = time.perf_counter()
        reply = json.loads(urllib.request.urlopen(request, timeout=600).read())["choices"][0]["message"]["content"] or ""
        seconds = time.perf_counter() - started
        self.cache[key] = reply
        with open(self.cache_path, "a") as fh:  # one line per reply; appends are atomic enough for a few threads
            fh.write(json.dumps({"key": key, "reply": reply}) + "\n")
        return reply, seconds


def parse(reply):
    """(verdict 1 / 0, or None when the reply has no readable verdict; the reasons list)."""
    found = re.findall(r"\{.*\}", reply, re.S)
    try:
        answer = json.loads(found[-1])
        verdict = {"tunnel": 1, "benign": 0}.get(str(answer.get("verdict", "")).lower())
        return verdict, answer.get("reasons", [])
    except (IndexError, json.JSONDecodeError):
        return None, []


def arbitrate(arbiter, rows, variant="full", workers=4):
    """Ask the LLM about every row. Returns a frame with verdict, parsed, seconds, reply."""

    def one(row):
        reply, seconds = arbiter.ask(SYSTEM, user_prompt(row, variant))
        verdict, _ = parse(reply)
        forest_vote = int(row.random_forest_score >= 0.5)
        return {"verdict": forest_vote if verdict is None else verdict, "parsed": verdict is not None,
                "seconds": seconds, "reply": reply}

    with ThreadPoolExecutor(workers) as pool:
        return pd.DataFrame(list(pool.map(one, [r for _, r in rows.iterrows()])), index=rows.index)


def counts(label, decision):
    tp = int(((label == 1) & (decision == 1)).sum()); fp = int(((label == 0) & (decision == 1)).sum())
    fn = int(((label == 1) & (decision == 0)).sum()); tn = int(((label == 0) & (decision == 0)).sum())
    return tn, fp, fn, tp


def rates(tn, fp, fn, tp):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "fnr": round(fn / (fn + tp), 4) if fn + tp else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="../output")
    parser.add_argument("--workers", type=int, default=4, help="requests sent to the model at the same time")
    parser.add_argument("--sample", type=int, default=20, help="escalated rows used for the prompt variants")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="ask the LLM about at most this many rows per case (the rows are in random test order); "
                             "the cascade numbers of a cut case are scaled up from those rows")
    args = parser.parse_args()
    tables = Path(args.out) / "tables"
    arbiter = Arbiter(Path(args.out) / "llm_cache.jsonl")
    cascade = pd.read_csv(tables / "cascade.csv")

    summary, cascade_rows, everything = [], [], []
    for train, test in CASES:
        rows = pd.read_csv(tables / f"escalated_{train}_{test}.csv")
        rows["subdomain"] = rows["subdomain"].fillna("")
        if rows.empty:
            continue
        total = len(rows)
        rows = rows.head(args.max_rows) if args.max_rows else rows
        started = time.time()
        result = arbitrate(arbiter, rows, workers=args.workers)
        label = rows["label"]
        forest, cnn = (rows["random_forest_score"] >= 0.5).astype(int), (rows["cnn_score"] >= 0.9).astype(int)
        llm = result["verdict"]
        summary.append({"train": train, "test": test, "escalated": total, "tunnels": int(label.sum()),
                        "llm_accuracy": round((llm == label).mean(), 4),
                        "forest_accuracy": round((forest == label).mean(), 4),
                        "cnn_accuracy": round((cnn == label).mean(), 4)})
        # The cascade with the LLM as stage 3: start from the cascade whose stage 3 is the forest's vote,
        # and swap the forest's vote on the escalated rows for the LLM's verdict.
        base = cascade[(cascade.train == train) & (cascade.test == test) &
                       (cascade.configuration == "cascade (stage 3 = random_forest)")].iloc[0]
        scale = total / len(rows)  # 1 unless the case was cut by --max-rows
        swap = [round((a - b) * scale) for a, b in zip(counts(label, llm), counts(label, forest))]
        cascade_rows.append({"train": train, "test": test, "configuration": "cascade (stage 3 = LLM)",
                             **rates(*(int(base[k]) + d for k, d in zip(("TN", "FP", "FN", "TP"), swap))),
                             "escalated": total, "estimated": scale != 1})
        everything.append(rows.assign(case=f"{train}_{test}", llm=llm, reply=result["reply"]))
        print(f"train {train}, test {test}: {len(rows)} rows in {time.time() - started:.0f} s, "
              f"LLM accuracy {summary[-1]['llm_accuracy']}, forest {summary[-1]['forest_accuracy']}, CNN {summary[-1]['cnn_accuracy']}")
    pd.DataFrame(summary).to_csv(tables / "llm_arbiter.csv", index=False)
    pd.DataFrame(cascade_rows).to_csv(tables / "llm_cascade.csv", index=False)

    # Prompt sensitivity on a fixed random sample of all escalated rows.
    pool = pd.concat(everything, ignore_index=True)
    sample = pool.sample(min(args.sample, len(pool)), random_state=SEED)
    reference = sample["llm"]
    sensitivity = []
    for variant in VARIANTS:
        verdict = reference if variant == "full" else arbitrate(arbiter, sample, variant, args.workers)["verdict"]
        sensitivity.append({"variant": variant, "rows": len(sample),
                            "verdicts_flipped": round((verdict != reference).mean(), 4),
                            "accuracy": round((verdict == sample["label"]).mean(), 4)})
        print(f"variant {variant}: flipped {sensitivity[-1]['verdicts_flipped']}, accuracy {sensitivity[-1]['accuracy']}")
    pd.DataFrame(sensitivity).to_csv(tables / "llm_prompt_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
