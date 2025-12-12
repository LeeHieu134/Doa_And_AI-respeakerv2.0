"""Metric helpers shared across training scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], average: str = "macro"
) -> Dict[str, float]:
    """Return common multi-class metrics as a dictionary."""
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def save_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: Iterable[str],
    out_path: Path,
) -> None:
    """Render and save a confusion matrix heatmap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: Iterable[str],
    out_path: Path,
) -> None:
    """Write sklearn classification report to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    out_path.write_text(report)


