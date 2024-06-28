from loan_approval_prediction.config import config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd


class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mean_dict = {}

    def fit(self, X, y=None):
        for var in self.variables:
            self.mean_dict[var] = X[var].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.mean_dict[var])
        return X


class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mode_dict = {}

    def fit(self, X, y=None):
        for var in self.variables:
            self.mode_dict[var] = X[var].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.mode_dict[var])
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X


class AddingVariables(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, ref_variable=None):
        self.variables = variables
        self.ref_variable = ref_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables] = X[self.variables] + X[self.ref_variable]
        return X


class LogTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 0.00000001
        for var in self.variables:
           X[var] = np.log(X[var] + eps)
        return X


class MyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = OrdinalEncoder(*args, **kwargs)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        cols = X.columns
        X = self.encoder.transform(X)
        encoded_df = pd.DataFrame(X, columns=cols)
        return encoded_df
