import numpy as np


class MinMaxScaler:
    def __init__(self, min_=0, max_=1, negate=False):
        self.min_ = min_
        self.max_ = max_
        self.negate = negate

    def fit_transform(self, x):
        x = np.array(x).astype(float)
        if self.negate:
            x = -1 * x
        diff = self.max_ - self.min_
        x -= min(x)
        x *= diff / max(x)
        x += self.min_
        return x
