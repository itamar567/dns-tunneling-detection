"""Stage 5, Evaluation. Works on any model's scores; knows nothing about the model or the dataset."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def metrics(y, score, threshold):
    """Confusion-matrix counts and the rates computed from them. A row is called a tunnel when score >= threshold."""
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "fnr": round(fn / (fn + tp), 4) if fn + tp else 0.0,
    }


def outcomes(y, score, threshold):
    """TN, FP, FN or TP for each row."""
    predicted = np.asarray(score) >= threshold
    actual = np.asarray(y) == 1
    return np.select([~actual & ~predicted, ~actual & predicted, actual & ~predicted], ["TN", "FP", "FN"], "TP")


def confusion_figure(results, path):
    """One confusion matrix per (model, dataset) in `results`, a list of dicts from metrics() with
    "model" and "dataset" keys added."""
    cols = min(4, len(results))
    rows = -(-len(results) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows), squeeze=False)
    for ax in axes.flat[len(results):]:
        ax.axis("off")
    for ax, r in zip(axes.flat, results):
        cells = [[r["TN"], r["FP"]], [r["FN"], r["TP"]]]
        ax.imshow([[0, 1], [1, 0]], cmap="RdYlGn_r", alpha=0.25)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cells[i][j]:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["benign", "tunnel"])
        ax.set_yticks([0, 1], ["benign", "tunnel"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(f"{r['model']}, dataset {r['dataset']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_table(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
