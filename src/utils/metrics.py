import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt


def ccc(x, y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc


if __name__ == '__main__':
    rng = default_rng()
    X = np.random.random_sample(1000)
    Y = np.zeros_like(X)

    sigma = 0.01
    tilt = 0
    for i in range(X.shape[0]):
        Y[i] = tilt*(X[i]-0.5) + rng.normal(X[i], sigma)

    print("CCC: %5.5f" % (ccc(X, Y)))
