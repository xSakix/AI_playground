import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import sigmoid_kernel, linear_kernel

np.warnings.filterwarnings('ignore')


def kernel_score(x, y):
    return sigmoid_kernel(x, y)


def my_score(x, y):
    return x[-1] - y[-1]


# can be changed to iterable
class State:
    def __init__(self, window, data):
        self.window = window
        pct = data.pct_change()
        self.bench = data.apply(lambda x: (x / data.iloc[0]) - 1.).as_matrix()

        self.pct = pct.as_matrix().reshape(-1)
        self.mean = pct.rolling(window=window).mean().as_matrix().reshape(-1)
        self.median = pct.rolling(window=window).median().as_matrix().reshape(-1)
        std = pct.rolling(window=window).std().as_matrix().reshape(-1)
        self.upperbb = self.mean + (2 * std)
        self.lowerbb = self.mean - (2 * std)

    def get_state(self, i):
        return np.array([self.pct[i],
                         self.lowerbb[i],
                         self.mean[i],
                         self.median[i],
                         self.upperbb[i]])

    def get_whole_state(self):
        return np.stack((self.pct,
                         self.lowerbb,
                         self.mean,
                         self.median,
                         self.upperbb), axis=1)

    def get_partial_state(self, t, window):
        return np.stack((self.pct[t - window:t],
                         self.lowerbb[t-window:t],
                         self.mean[t-window:t],
                         self.median[t-window:t],
                         self.upperbb[t-window:t]), axis=1)
