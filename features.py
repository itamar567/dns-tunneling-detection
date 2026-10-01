"""The final feature list and how each feature is computed from a query name."""
import math
from collections import Counter, deque

import numpy as np

FINAL = ["subdomain_len", "qname_entropy", "label_count", "label_max", "label_avg", "same_sld_count"]
FINAL_B_ONLY = ["qtype"]
QTYPE_COLUMNS = ["qtype_A", "qtype_CNAME", "qtype_TXT", "qtype_NULL", "qtype_other"]
# Fixed divisors from the DNS protocol limits
SCALE = {"subdomain_len": 253, "qname_entropy": 6, "label_count": 127,
         "label_max": 63, "label_avg": 63}  # same_sld_count is divided by its window, see preprocess.scale
# Reverse-lookup suffixes occupy two labels, so the second-level domain is one label further left.
TWO_LABEL_SUFFIXES = {("in-addr", "arpa"), ("ip6", "arpa")}
# Country second-level labels, as in bbc.co.uk
COUNTRY_SECOND_LEVEL = {"co", "com", "net", "org", "gov", "edu", "ac", "or", "ne", "go", "gob", "mil", "nic"}


def shannon_entropy(text):
    """Shannon entropy in bits of the characters of a string."""
    if not text:
        return 0.0
    counts = Counter(text).values()
    return -sum(c / len(text) * math.log2(c / len(text)) for c in counts)


def split_domain(qname):
    """Split a query name into (all labels, subdomain labels, second-level domain)."""
    labels = str(qname).strip().rstrip(".").split(".") if str(qname).strip(".") else []
    two_part = tuple(labels[-2:]) in TWO_LABEL_SUFFIXES or (
        len(labels) >= 2 and labels[-2] in COUNTRY_SECOND_LEVEL and len(labels[-1]) == 2)
    suffix = 2 if two_part else 1
    if len(labels) > suffix:
        return labels, labels[:-(suffix + 1)], labels[-(suffix + 1)]
    return labels, [], (labels[0] if labels else "")


def lexical_features(qname):
    """Per-query features of one domain name, following the CIC extractor's definitions."""
    labels, sub_labels, sld = split_domain(qname)
    return {
        "qname_len": len(".".join(labels)),
        "subdomain": ".".join(sub_labels),
        "subdomain_len": len(".".join(sub_labels)),
        "qname_entropy": shannon_entropy(".".join(labels)),
        "label_count": len(labels),
        "label_max": max((len(label) for label in labels), default=0),
        "label_avg": sum(len(label) for label in labels) / len(labels) if labels else 0.0,
        "sld": sld,
    }


def same_sld_count(slds, window=100):
    """For each row, how many of the previous `window` rows queried the same second-level domain."""
    counts, recent, out = Counter(), deque(), []
    for sld in slds:
        out.append(counts[sld])
        counts[sld] += 1
        recent.append(sld)
        if len(recent) > window:
            counts[recent.popleft()] -= 1
    return out


def feature_matrix(df, with_qtype):
    """X (the six final features, plus the one-hot query type columns if with_qtype) and y as arrays."""
    columns = FINAL + (QTYPE_COLUMNS if with_qtype else [])
    return df[columns].to_numpy(dtype=np.float32), df["label"].to_numpy()
