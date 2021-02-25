import numpy as np

class Cloud():
    def __init__(self, n_neighbors="auto", metric="precomputed", percentile_cutoff=0.95):
        """
        n_neighbors: Interger/String default='auto'
            'auto' will set k = 10% of reference set size
        """
        self.k = n_neighbors
        self.metric=metric
        self.percentile_cutoff = percentile_cutoff
        
    def fit(self, X):
        """
        x: 2-d np.array
            Distances between each sample in the reference set. The distance between a
            sample and itself should be zero. Note: X should be square
            where row i represents sample i and column j represents sample j. So,
            X[i,j] should be the distance between sample i and j and X[i, i] = 0.
        """
        if self.k == "auto":
            k = int(X.shape[0] * 0.1) 
            self.k = k if k > 0 else 1
        
        if self.metric == "precomputed":
            self.di = np.apply_along_axis(
                lambda x: x[np.argpartition(x, self.k+1)[:self.k+1]].sum() / self.k,
                axis=1,
                arr=X
            )

        self.d_bar = self.di.mean()
        self.ri = self.di / self.d_bar
    
    def score_samples(self, X):
        """
        X: 2-d np.array
            Distance between test samples and reference samples. Each row in X should represent a test sample
            and each column should represent a reference sample. So, X[i,j] is the distance between test sample
            i and reference sample j.
        """
        dj = np.apply_along_axis(
            lambda x: x[np.argpartition(x, self.k)[:self.k]].mean(),
            axis=1,
            arr=X
        )
        rj = dj / self.d_bar
        return rj
        
    def decision_function(self, X):
        """
        X: 2-d np.array
            Distance between test samples and reference samples. Each row in X should represent a test sample
            and each column should represent a reference sample. So, X[i,j] is the distance between test sample
            i and reference sample j.
        """
        scores = self.score_samples(X)
        return -1*(scores - np.quantile(self.ri, self.percentile_cutoff))
    
    def predict(self, X):
        """
        X: 2-d np.array
            Distance between test samples and reference samples. Each row in X should represent a test sample
            and each column should represent a reference sample. So, X[i,j] is the distance between test sample
            i and reference sample j.
        """
        df = self.decision_function(X)
        df[df > 0] = 1
        df[df <= 0] = -1
        return df
        