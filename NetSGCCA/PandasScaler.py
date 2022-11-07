from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import pandas as pd

class PandasScaler(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
    def transform(self, X, y=None):
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
