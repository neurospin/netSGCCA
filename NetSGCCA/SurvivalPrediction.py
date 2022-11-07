from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.metrics import integrated_brier_score


class SurvivalPrediction(BaseEstimator, RegressorMixin):
    def __init__(self, lr=1e-1, l2_reg=0, init_method='zeros', max_iter=10000, tol=1e-5, event="status",
                 event_time="time"):
        self.coxph = CoxPHModel(auto_scaler=True)
        self.lr = lr
        self.l2_reg = l2_reg
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.event = event
        self.event_time = event_time

    def fit(self, X, y):
        self.coxph.fit(X, y.loc[:, self.event_time], y.loc[:, self.event], lr=self.lr, l2_reg=self.l2_reg,
                       init_method=self.init_method,
                       max_iter=self.max_iter, tol=self.tol, verbose=False)
        return self

    def score(self, X, y, scoring='concordance', *args, **kwargs):
        scores = {}
        if scoring == "concordance" or scoring == "both":
            score_c = concordance_index(self.coxph, X, y.loc[:, self.event_time], y.loc[:, self.event])
            scores["concordance"] = score_c
        if scoring == "brier" or scoring == "both":
            score_b = -integrated_brier_score(self.coxph, X, y.loc[:, self.event_time], y.loc[:, self.event])
            scores["brier"] = score_b
        return score_c
