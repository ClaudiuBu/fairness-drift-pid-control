import numpy as np

def demographic_parity(y_pred, A):
    """
    P(y=1 | A=1) - P(y=1 | A=0)
    """
    p1 = y_pred[A == 1].mean()
    p0 = y_pred[A == 0].mean()
    return p1 - p0


def precision_score(y_true, y_pred):
    """Precision: TP / (TP + FP)"""
    if y_pred.sum() == 0:
        return 0.0
    return (y_true[y_pred == 1] == 1).sum() / y_pred.sum()


def recall_score(y_true, y_pred):
    """Recall: TP / (TP + FN)"""
    if y_true.sum() == 0:
        return 0.0
    return (y_pred[y_true == 1] == 1).sum() / y_true.sum()


def f1_score(y_true, y_pred):
    """F1: Harmonic mean of precision and recall"""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)