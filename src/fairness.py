import numpy as np

def demographic_parity(y_pred, A):
    """
    P(y=1 | A=1) - P(y=1 | A=0)
    """
    p1 = y_pred[A == 1].mean()
    p0 = y_pred[A == 0].mean()
    return p1 - p0