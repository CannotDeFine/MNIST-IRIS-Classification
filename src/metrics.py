"""Classification metrics implemented with NumPy."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Return a confusion matrix with rows as true labels and columns as predictions."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    """Return per-class accuracy from a confusion matrix."""
    matrix = np.asarray(matrix)
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    return np.divide(correct, totals, out=np.zeros_like(correct, dtype=np.float64), where=totals != 0)
