import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Hyper parameters
K = 2
M = 2
N = 100


# def kmean_serial():

def kpp_serial(X, k):
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
        print("C[j] selected:", C[j])
    return C


def generate_data():
    data = np.random.rand(N, M)
    return data


def weighted_rand_index(W):
    r = np.sum(W) * np.random.rand(1)[0]
    i = 0
    s = W[0]
    while s < r:
        i += 1
        s += W[i]
    return i


def plot_kmeans_pp(x, c):
    if x.shape[1] != 2 or c.shape[1] != 2:
        raise ValueError("M must be 2 dimensions to plot.")
    plt.scatter(x[:, 0], x[:, 1], color='black')
    plt.scatter(c[:, 0], c[:, 1], color='red')
    plt.show()


if __name__ == "__main__":
    X = generate_data()
    C = kpp_serial(X, K)
    plot_kmeans_pp(X, C)
