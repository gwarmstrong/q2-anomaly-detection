import tempfile
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin
from skbio.stats.composition import clr
from skbio.stats import subsample_counts
from skbio.diversity.beta import unweighted_unifrac
from q2_anomaly_detection.utils import as_dense
from functools import partial
import pandas as pd
from unifrac import ssu


class AsDense(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return as_dense(X)


class UniFrac(TransformerMixin):

    def __init__(self, tree, **kneighbors_kwargs):
        self.tree = tree
        self.otu_ids = None
        self.distance_fn = None
        self.input_data = None

    def fit(self, X, y=None):
        """

        X : pd.DataFrame
            Must contain the OTU IDs in the column names

        """
        self.otu_ids = X.columns.values
        self.distance_fn = partial(unweighted_unifrac, otu_ids=self.otu_ids,
                                   tree=self.tree,
                                   )
        # TODO this may want to make a copy of X
        self.input_data = X
        return self

    def transform(self, X):
        return cdist(X, self.input_data, metric=self.distance_fn)


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
        if isinstance(X, pd.DataFrame):
            idx = np.array([True] * len(X.columns))
            idx[self.idx[:, 1]] = False
            X = X.loc[:, idx]
        else:
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
        iter_var = X.values if isinstance(X, pd.DataFrame) else X
        for row in iter_var:
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
