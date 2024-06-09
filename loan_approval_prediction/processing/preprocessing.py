from loan_approval_prediction.config import config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables
        self.mean_dict = {}

    def fit(self, X, y=None):
        for var in self.variables:
            self.mean_dict[var] = X[var].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.mean_dict[var], inplace=True)
        return X


class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.mean_dict = {}

    def fit(self, X, y=None):
        for var in self.variables:
            self.mean_dict[var] = X[var].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.mean_dict[var], inplace=True)
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.drop(self.variables, axis=1, inplace=True)
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
        for var in self.variables:
            X[var] = np.log(X[var])
        return X

class MyLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)
