

"""
Helper utilities for modeling
"""

from collections import OrderedDict
from functools import partial
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, recall_score, precision_score, fbeta_score, roc_auc_score, classification_report)
from sklearn.metrics import roc_auc_score


class Baseline(ClassifierMixin, BaseEstimator):
    """
    Baseline model based on the lookup approach
    """

    def __init__(self, *, best_features=None, threshold=None):
        self.best_features = best_features or ['status', 'history', 'savings']
        self.threshold = threshold or 0.5
    
    def fit(self, df, y=None):
        if y is not None:
            df = df.assign(label=y)
        cols = list(self.best_features) + ['label']
        lookup = df[cols].groupby(cols[:-1]).mean().reset_index()
        #lookup['label'] = lookup['label'].fillna(value=df['label'].mean())
        lookup.rename(columns={'label': "probability"}, inplace=True)
        self.lookup_ = lookup
        self.prior_ = df['label'].mean()
        self.classes_ = np.array([0, 1])
        return self
    
    def predict_proba(self, df):
        probs = list()
        for i,sr in df.iterrows():
            query = " & ".join([f"{s} == '{sr[s]}'" for s in self.best_features])
            df_query = self.lookup_.query(query)
            probs.append(df_query.iloc[0,-1] if len(df_query) else self.prior_)
        return np.array(probs)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(np.uint8)



def classification_scores(y_true, y_pred, round=True) -> pd.Series:
    funcs = OrderedDict([
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", partial(fbeta_score, beta=1)),
        ("f2", partial(fbeta_score, beta=2)),
    ])
    sr = pd.Series({k:f(y_true, y_pred) for k,f in funcs.items()})
    return sr.round(int(2 if round is True else round)) if round else sr



def f2_scorer(*args):
    """
    Works both as a scorer e.g. for cross_val_score -> implicit signature: (estimator, X, y)
    and 'f2_score'  -> implicit signature: (y_true, y_pred)
    """
    if len(args) == 2:
        return fbeta_score(*args, beta=2)
    est, X, y = args
    return fbeta_score(y, est.predict(X), beta=2)


def auc_scorer(*args):
    """
    analogoues to f2_scorer
    """
    if len(args) == 2:
        return roc_auc_score(*args)
    clf, X, y = args
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])