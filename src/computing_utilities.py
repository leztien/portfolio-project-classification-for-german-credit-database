

"""
Computing utilities
"""


import numpy as np
from sklearn.metrics import fbeta_score
from functools import partial
from sklearn.model_selection import train_test_split



def optimize_threshold(model, X, y, val_size=0.2, metric='f2', random_state=None):
    """
    Threshold optimization
    Returns (best_threshold, best_metric)
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    metric = partial(fbeta_score, beta=2) if str(metric).lower() == 'f2' else metric

    probs = model.fit(X_train, y_train).predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0, 1, 100)
    scores = [metric(y_val, (probs >= thresh).astype(int)) for thresh in thresholds]

    return thresholds[np.argmax(scores)], max(scores)




def error_sets_difference(y_true, y_pred_a, y_pred_b, error_type=None):
    y_true, y_pred_a, y_pred_b = (np.array(a) for a in (y_true, y_pred_a, y_pred_b))

    if str(error_type).upper() == 'FN':
        a = (y_true == 1) & (y_pred_a == 0)
        b = (y_true == 1) & (y_pred_b == 0)

    elif str(error_type).upper() == 'FP':
        a = (y_true == 0) & (y_pred_a == 1)
        b = (y_true == 0) & (y_pred_b == 1)
    else:
        a = y_true != y_pred_a
        b = y_true != y_pred_b

    similariry = sum(a & b) / sum(a | b)
    return 1 - similariry