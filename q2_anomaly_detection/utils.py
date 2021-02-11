def as_dense(biom_table):
    return biom_table.matrix_data.todense().transpose()


class IdentityScaler:
    def __call__(self, x):
        return x