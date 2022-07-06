import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


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


    with plt.style.context(('seaborn-whitegrid')):
        plt.figure(figsize=(8, 6))

        # Scatter plot of X vs Y
        plt.scatter(X, Y, edgecolors='k', alpha=0.5)

        # Plot of the 45 degree line
        plt.plot([0, 1], [0, 1], 'r')

        plt.text(0, 0.75*Y.max(), "CCC: %5.5f" % (ccc(X, Y)),
                fontsize=16, bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.8, 0.1, "$\sigma=$ %5.3f" % (sigma)+"\nTilt = %5.3f" % (tilt),
                fontsize=16, bbox=dict(facecolor='white', alpha=0.5))

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)

        plt.show()
