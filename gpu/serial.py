import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Hyper parameters
M = 5
K = 2
N = 10

def generate_data():
    data = np.random.randn(N, M)
    # data = np.random.rand(N, M)
    return data

def kpp_serial(X, K):
    # n = X.shape[0]
    # m = X.shape[1]
    D = np.zeros(N)
    for i in range(N):
        D[i] = np.inf

    C = np.zeros((K, M))

    # The first seed is selected uniformly at random
    index = np.random.randint(N)
    C[0] = X[index]

    for j in range(1, K):
        for i in range(N):
            # Update the nearest distance, if necessary
            D[i] = min(np.linalg.norm(X[i] - C[j - 1]), D[i])
        i = weighted_rand_index(D)
        C[j] = X[i]
        print("C[{}] selected: {}".format(j,  C[j]))
    return C

def weighted_rand_index(W):
    r = np.sum(W) * np.random.rand(1)[0]
    i = 0
    s = W[0]
    while s < r:
        i += 1
        s += W[i]
    return i


def plot_kmeans_pp():
    if X.shape[1] != 2 or C.shape[1] != 2:
        raise ValueError("M must be 2 dimensions to plot.")
    plt.scatter(X[:, 0], X[:, 1], color='black')
    plt.scatter(C[:, 0], C[:, 1], color='red')
    plt.show()


def plot_kmeans():
    if X.shape[1] != 2 or C.shape[1] != 2:
        raise ValueError("M must be 2 dimensions to plot.")
    km = KMeans(n_clusters=K, init=C, n_init=1)
    labels = km.fit_predict(X)
    rancols = [np.random.rand(3).tolist() for _ in range(max(labels) + 1)]
    colmap = dict(zip(range(max(labels) + 1), rancols))
    colors = [colmap[l] for l in labels]
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c=colors[i], s=3)
    plt.show()

def compare_kmeans(nruns=1000):
    initnames = ["ours", "sklearn", "random"]
    for i, init in enumerate([C, 'k-means++', 'random']):
        iters = []
        for run in range(nruns):
            # print("Run {} in {}".format(run, init))
            km = KMeans(n_clusters=K, init=init, n_init=1)
            km.fit_predict(X)
            iters.append(km.n_iter_)
        print("Average {} n_iters needed: {}".format(initnames[i], np.mean(iters)))


if __name__ == "__main__":
    X = generate_data()
    print("X is: \n", X)
    C = kpp_serial(X, K)

    print("C is \n", C)
    print(C)
    # compare_kmeans()
    # plot_kmeans_pp()
    # plot_kmeans()
