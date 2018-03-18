# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


def load_planar(n_samples=1000):
    # Generate data
    np.random.seed(1)
    m = n_samples  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y


def load_circles(n_samples=1000):
    np.random.seed(1)
    X, y = sklearn.datasets.make_circles(n_samples=n_samples, noise=.05)

    # Visualize the data
    # plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    # plt.show()
    return X, y