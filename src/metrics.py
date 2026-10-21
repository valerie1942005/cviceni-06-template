from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Sestavi 2x2 kontingecni tabulku [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert y_true.shape == y_pred.shape, "Pole y_true a y_pred musí mít stejnou délku."
    assert set(np.unique(y_true)).issubset({0, 1}), "y_true smí obsahovat pouze hodnoty {0, 1}."
    assert set(np.unique(y_pred)).issubset({0, 1}), "y_pred smí obsahovat pouze hodnoty {0, 1}."

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ], dtype=int)


def accuracy(cm: np.ndarray) -> float:
    """Presnost (accuracy) — podil spravne zaradenych vzorku."""
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)."

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    denominator = tp + tn + fp + fn

    return float((tp + tn) / denominator) if denominator > 0 else 0.0


def precision(cm: np.ndarray) -> float:
    """Preciznost (precision, pozitivni prediktivni hodnota)."""
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)."

    fp, tp = cm[0, 1], cm[1, 1]
    denominator = tp + fp

    return float(tp / denominator) if denominator > 0 else 0.0


def recall_sensitivity(cm: np.ndarray) -> float:
    """Recall = sensitivita = Se = TPR (true positive rate)."""
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)."

    fn, tp = cm[1, 0], cm[1, 1]
    denominator = tp + fn

    return float(tp / denominator) if denominator > 0 else 0.0


def specificity(cm: np.ndarray) -> float:
    """Specificita = Sp = TNR (true negative rate)."""
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)."

    tn, fp = cm[0, 0], cm[0, 1]
    denominator = tn + fp

    return float(tn / denominator) if denominator > 0 else 0.0


def f1_score(cm: np.ndarray) -> float:
    """F1 skore — harmonicky prumer preciznosti a recallu."""
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)."

    fp, fn, tp = cm[0, 1], cm[1, 0], cm[1, 1]
    denominator = 2 * tp + fp + fn

    return float((2 * tp) / denominator) if denominator > 0 else 0.0
