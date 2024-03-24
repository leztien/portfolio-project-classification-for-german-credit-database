

"""
Helper utilities for modeling
"""

from collections import OrderedDict
from functools import partial
from functools import partial
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, recall_score, precision_score, fbeta_score, roc_auc_score, classification_report)
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline



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



def _make_ratio(X):
    return X[:, [0]] / X[:, [1]]


def make_ratio_pipline(imputer_strategy='median', log=False):
    steps = [
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("ratio", FunctionTransformer(_make_ratio)),
        ("log", FunctionTransformer(np.log)),
        ("scaler", StandardScaler())]

    if not log:
        del steps[-2]
    return Pipeline(steps)



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