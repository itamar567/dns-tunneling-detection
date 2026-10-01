"""Stage 1, Data Ingestion: the only file that knows the datasets' own layout.

Every loader returns one table with exactly the columns in COLUMNS, so nothing
downstream needs to know which dataset it is looking at.

  Dataset A: data/processed/cic-bell-dns-exf-2021/names/**/<capture>.csv (see extract_names.py)
  Dataset B: data/netrack-learn/dns/<tool>/<timestamp>.csv

`subdomain` is the text before the second-level domain.
`truncated` is 1 when the capture cut the query name short; see restore_truncated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from features import FINAL, FINAL_B_ONLY, lexical_features, same_sld_count

COLUMNS = FINAL + FINAL_B_ONLY + ["qname_len", "sld", "qname", "subdomain", "truncated", "label", "capture"]

ATTACK_DOMAIN = "cicresearch.ca"  # every DNSExfiltrator query in Dataset A went to this domain (CIC paper)
CHUNK = 63                        # and each data label was filled to the DNS maximum
LABEL_BY_CONTENT = True           # False: every row of an attack capture is an attack, as in the CIC files


def _from_names(qnames, qtypes, label, capture, truncated=None, exact_len=None):
    """Build the unified table from raw query names, the same way for every dataset."""
    df = pd.DataFrame([lexical_features(q) for q in qnames.astype(str)])
    if exact_len is not None:
        df["qname_len"] = exact_len.fillna(df["qname_len"]).astype(int).values
    df["qname"], df["qtype"] = qnames.values, qtypes.values
    df["truncated"] = 0 if truncated is None else truncated.values
    df["label"], df["capture"] = label, capture
    return df


def restore_truncated(df):
    """The captures cut long names after about 40 characters. A cut-off attack name was
    <counter>.<63-character chunks>.cicresearch.ca and its exact length is known, so its
    structural features follow by arithmetic. The few cut-off benign names cannot be
    restored and are dropped."""
    df = df[(df["truncated"] == 0) | (df["label"] == 1)].copy()
    cut = df["truncated"] == 1
    length = df.loc[cut, "qname_len"]
    counter = df.loc[cut, "qname"].str.split(".").str[0].str.len()
    data = length - counter - 1 - len(ATTACK_DOMAIN) - 1        # the chunks and the dots between them
    chunks = np.ceil((data + 1) / (CHUNK + 1))
    labels = 1 + chunks + 2                                     # counter, chunks, cicresearch, ca
    df.loc[cut, "subdomain"] = df.loc[cut, "qname"]             # everything captured is subdomain
    df.loc[cut, "subdomain_len"] = length - len(ATTACK_DOMAIN) - 1
    df.loc[cut, "label_count"] = labels
    df.loc[cut, "label_max"] = np.minimum(CHUNK, data)
    df.loc[cut, "label_avg"] = (length - labels + 1) / labels
    df.loc[cut, "sld"] = ATTACK_DOMAIN.split(".")[0]
    return df


def load_dataset_a(names_dir, window=100):
    """Dataset A: the per-capture CSVs written by extract_names.py. A capture is benign when its
    file name contains "benign", the same rule the CIC feature files follow. Inside an attack
    capture only the DNSExfiltrator queries are attacks: the victim's own background traffic
    (dns.google, Microsoft telemetry, ...) is recorded there too and is benign."""
    paths = sorted(Path(names_dir).rglob("*.csv"))
    assert len({p.stem for p in paths}) == len(paths), "capture names are not unique"
    parts = []
    for path in paths:
        raw = pd.read_csv(path, low_memory=False)
        raw["qname"] = raw["qname"].fillna("")
        label = 0 if "benign" in path.stem else 1
        df = _from_names(raw["qname"], raw["qtype"], label, path.stem, truncated=raw["truncated"], exact_len=raw["qname_len"])
        df = restore_truncated(df)
        if LABEL_BY_CONTENT:
            df["label"] = label * (df["sld"] == ATTACK_DOMAIN.split(".")[0])
        df["same_sld_count"] = same_sld_count(df["sld"], window)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_dataset_b(netrack_dir, window=100):
    """Dataset B: the netrack/learn repository. One packet capture per tool folder under dns/
    (names.csv next to it is a name list, not packets). `capture` is the tool; plain is benign."""
    parts = []
    for path in sorted(Path(netrack_dir).glob("dns/*/*.csv")):
        if path.name == "names.csv":
            continue
        raw = pd.read_csv(path, low_memory=False)
        tool = path.parent.name
        df = _from_names(raw["qname"], raw["qd_qtype"], 0 if tool == "plain" else 1, tool)
        df["same_sld_count"] = same_sld_count(df["sld"], window)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


LOADERS = {"A": load_dataset_a, "B": load_dataset_b}


def load(name, path, window=100):
    """Load one dataset and check that it has exactly the unified columns (the dependency rule)."""
    df = LOADERS[name](path, window)
    missing, extra = set(COLUMNS) - set(df.columns), set(df.columns) - set(COLUMNS)
    assert not missing and not extra, f"{name}: missing {missing}, extra {extra}"
    return df[COLUMNS]


def deduplicate(df):
    """Drop exact repeats of the feature vector within a class."""
    return df.drop_duplicates(subset=FINAL + FINAL_B_ONLY + ["label"])
