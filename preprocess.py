from sklearn.model_selection import StratifiedKFold, train_test_split

from common import SEED, WINDOW
from ingest import deduplicate
from features import SCALE

QTYPES = {1: "A", 5: "CNAME", 16: "TXT", 10: "NULL"}


def split(df, test_size=0.3):
    """Deduplicate, then a stratified train/test split. Returns (train, test)."""
    df = deduplicate(df)
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def folds(train, k=5):
    """Stratified k-fold (train_index, validation_index) pairs over the rows of `train`."""
    return list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(train, train["label"]))


def scale(df, window=WINDOW):
    """Divide each feature by its DNS protocol limit, and same_sld_count
    by the window it was counted over."""
    df = df.copy()
    for feature, limit in {**SCALE, "same_sld_count": window}.items():
        df[feature] = df[feature] / limit
    return df


def unscale(df, window=WINDOW):
    """Undo scale(), back to the original units."""
    df = df.copy()
    for feature, limit in {**SCALE, "same_sld_count": window}.items():
        df[feature] = df[feature] * limit
    return df


def encode_qtype(df):
    """Turn the query type into five yes/no columns: qtype_A, qtype_CNAME, qtype_TXT, qtype_NULL and
    qtype_other. Each row gets True in the column of its type and False in the rest."""
    df = df.copy()
    names = df["qtype"].map(QTYPES).fillna("other")
    for name in [*QTYPES.values(), "other"]:
        df[f"qtype_{name}"] = (names == name)
    return df


def prepare(df, with_qtype, window=WINDOW):
    """split, then scale, then the query type one-hot when with_qtype. Returns (train, test)."""
    train, test = split(df)
    train, test = scale(train, window), scale(test, window)
    if with_qtype:
        train, test = encode_qtype(train), encode_qtype(test)
    return train, test


def class_weights(y):
    """Weight per class so both classes count the same in total."""
    counts = y.value_counts()
    return {label: len(y) / (2 * count) for label, count in counts.items()}
