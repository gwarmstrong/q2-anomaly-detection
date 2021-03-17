from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


def as_dense(biom_table):
    table = np.asarray(biom_table.matrix_data.todense().transpose())
    features = biom_table.ids('observation')
    sample_ids = biom_table.ids('sample')
    df = pd.DataFrame(table, columns=features, index=sample_ids)
    return df


class IdentityScaler:
    def __call__(self, x):
        return x


class TrivialScorer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def score_samples(self, X):
        return np.zeros(len(X))
