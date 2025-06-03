import pandas as pd
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import data_correction.dc1 as dc1


class TransformerOnTrainAndTest(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        df = X

        df = df.drop(columns=["WS_HR_num", "NOx_num", "PM2_5_num", "WD_HR_num"])

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class TransformerOnTrain(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        df = X

        self.is_fitted_ = True
        return df

    def transform(self, X, y=None):

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
