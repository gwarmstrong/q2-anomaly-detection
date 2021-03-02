import numpy as np


class Cloud():
    def __init__(
        self,
        n_neighbors="auto",
        metric="precomputed",
        percentile_cutoff=0.95
    ):
        """
        n_neighbors: "auto" or int default="auto"
            "auto" will set k = 10% of reference set size
        """
        self.k = n_neighbors
        self.metric = metric
        self.percentile_cutoff = percentile_cutoff

    def fit(self, X, y=None):
        """
        X: np.array of shape (n_samples, n_features)
            Distances between each sample in the reference set. The distance
            between a sample and itself should be zero. Note: X should be
            square where row i represents sample i and column j represents
            sample j. So, X[i,j] should be the distance between sample i and j
            and X[i, i] = 0.
        """
        if self.k == "auto":
            k = int(X.shape[0] * 0.1)
            self.k = k if k > 0 else 1

        if self.metric == "precomputed":
            _, neighbor_distances = self.kneighbors(X, n_neighbors=self.k+1)
            self.estimators_diameters = neighbor_distances[:, 1:].mean(axis=1)

        self.mean_diameter = self.estimators_diameters.mean()
        self.outlier_detection_test_scores = \
            self.estimators_diameters / self.mean_diameter

    def kneighbors(self, X=None, n_neighbors=None):
        n_neighbors = self.k if n_neighbors is None else n_neighbors
        indices = np.argpartition(X, kth=n_neighbors, axis=1)[:, :n_neighbors]
        distances = np.take_along_axis(X, indices, axis=1)
        return indices, distances

    def score_samples(self, X, y=None):
        """
        X: np.array of shape (n_samples, n_features)
            Distance between test samples and reference samples. Each row in X
            should represent a test sample and each column should represent a
            reference sample. So, X[i,j] is the distance between test sample
            i and reference sample j.
        """
        _, neighbor_distances = self.kneighbors(X)
        sample_diameters = neighbor_distances.mean(axis=1)
        sample_outlier_detection_test_scores = \
            sample_diameters / self.mean_diameter
        return -sample_outlier_detection_test_scores

    def decision_function(self, X):
        """
        X: np.array of shape (n_samples, n_features)
            Distance between test samples and reference samples. Each row in X
            should represent a test sample and each column should represent a
            reference sample. So, X[i,j] is the distance between test sample
            i and reference sample j.
        """
        scores = self.score_samples(X)
        return scores + np.quantile(
            self.outlier_detection_test_scores, self.percentile_cutoff
        )

    def predict(self, X):
        """
        X: np.array of shape (n_samples, n_features)
            Distance between test samples and reference samples. Each row in X
            should represent a test sample and each column should represent a
            reference sample. So, X[i,j] is the distance between test sample
            i and reference sample j.
        """
        df = self.decision_function(X)
        return np.where(df > 0, 1, -1)
