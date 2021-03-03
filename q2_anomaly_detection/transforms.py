import numpy as np
from sklearn.base import TransformerMixin
from skbio.stats.composition import clr
from skbio.stats import subsample_counts


class Rarefaction(TransformerMixin):

    def __init__(self, depth, replace=False):
        self.depth = depth
        self.replace = replace
        self.idx = None

    def fit(self, X, y=None):
        X, self.idx = self._find_nonzero_idx(X)
        return self

    def transform(self, X, y=None):
        """
        Caution: this will return different results for the same sample
        """
        X = np.delete(X, self.idx, axis=1)
        X = self._subsample(X)

        return X

    def _find_nonzero_idx(self, X):
        X = self._subsample(X)
        # remove columns with zero counts
        row_sums = X.sum(axis=0, keepdims=True)
        idx = np.argwhere(row_sums == 0)
        return X, idx

    def _subsample(self, X):
        X = X.astype(int)
        X_out = list()
        for row in X:
            new_X = subsample_counts(row, n=self.depth, replace=self.replace)
            X_out.append(new_X)
        X = np.vstack(X_out)
        return X


class CLR(TransformerMixin):

    def __init__(self, pseudocount=1):
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return clr(X + self.pseudocount)
