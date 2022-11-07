from sklearn.base import BaseEstimator, TransformerMixin
from lifelines import CoxPHFitter
import pandas as pd


class CoxResiduals(BaseEstimator, TransformerMixin):
    def __init__(self, penalizer=0, l1_ratio=0, event="status", event_time="time"):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.event_time = event_time
        self.event = event

    def fit(self, X=None, y=None):
        self.fitter.fit(y, duration_col=self.event_time, event_col=self.event)
        r = self.fitter.compute_residuals(y, 'deviance')
        self.residuals = r.deviance[y.index].to_frame()
        return self

    def transform(self, X, y=None):
        return X.join(self.residuals, how='left')
