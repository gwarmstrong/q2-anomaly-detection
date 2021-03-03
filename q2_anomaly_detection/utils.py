import numpy as np


def as_dense(biom_table):
    return np.asarray(biom_table.matrix_data.todense().transpose())


class IdentityScaler:
    def __call__(self, x):
        return x
