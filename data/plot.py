import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import random

# Hyper parameters
M = 2
K = 5
N = 1000

def generate_data():
	# data = np.random.randn(N, M)
	data = np.random.rand(N, M)
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


def plot_kmeans_pp(s=3):
	print(C.shape)
	if X.shape[1] != 2 or C.shape[1] != 2:
		raise ValueError("M must be 2 dimensions to plot.")
	plt.scatter(X[:, 0], X[:, 1], color='black', s=s)
	plt.scatter(C[:, 0], C[:, 1], color='red', s=s)
	plt.title("k-means++ initial seeding of 1000 points into 5 clusters")
	plt.show()


def plot_kmeans(s=3):
	if X.shape[1] != 2 or C.shape[1] != 2:
		raise ValueError("M must be 2 dimensions to plot.")
	km = KMeans(n_clusters=K, init=C, n_init=1)
	labels = km.fit_predict(X)
	rancols = [np.random.rand(3).tolist() for _ in range(max(labels) + 1)]
	colmap = dict(zip(range(max(labels) + 1), rancols))
	colors = [colmap[l] for l in labels]
	for i in range(X.shape[0]):
		plt.scatter(X[i][0], X[i][1], c=colors[i], s=s)
	plt.title("k-means clustered dataset")
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

def rancolor():
	r = lambda: random.randint(0,255)
	return '#%02X%02X%02X' % (r(),r(),r())

def omp_strong(comp, n, m, k, alt_title=None):
	df = pd.read_csv('omp_{}.csv'.format(comp))

	df = df[df['n'] == n]

	for i, mi in enumerate(m):
		for j, kj in enumerate(k):
			d = df[(df['m'] == mi) & (df['k'] == kj)]
			c = rancolor()
			print(d)
			plt.scatter(d['p'], d['t'], color=c, label=None)
			plt.plot(d['p'], d['t'], color=c, label='m = {}, k = {}'.format(mi, kj))

	compname = "Knight's Landing" if comp=="knl" else "Haswell"
	if alt_title:
		plt.title(alt_title)
	else:
		plt.title("OpenMP strong scaling on {} with N={}".format(compname, int(n)))
	plt.xlabel("Num Threads")
	plt.ylabel("Wall time(s)")
	plt.yscale('log')
	plt.xscale('log')
	plt.legend(loc='lower right')
	# plt.show()
	plt.savefig('strongtrends_{}.png'.format(comp))


def omp_weak(m, p=32):
	for comp in ['hsw', 'knl']:
		basecolor = 'green' if comp=='hsw' else 'blue'
		colors = ['', 'dark', 'light']
		for i, k in enumerate([2, 5, 10]):
			df = pd.read_csv('omp_{}.csv'.format(comp))

			df = df[(df['m'] == m) & (df['k'] == k)]
			serial = df[df['p'] == 2]['t'] * 2.0
			n = [1e3, 1e4, 1e5, 1e6]
			# print(df)

			c = colors[i] + basecolor
			print(c)
			d = df[df['p'] == p]
			print(d)
			print(serial)
			speedup = serial.as_matrix()[:len(d['t'])]/(d['t'].as_matrix())
			# print(speedup)
			lab = "Knight's Landing" if comp == 'knl' else "Haswell".format()
			plt.plot(n[:len(d['t'])], speedup, color=c, label=lab + ' k={}'.format(k))
			plt.scatter(n[:len(d['t'])], speedup, color=c)

	plt.title("Speedup vs. N, M={} for KNL (68 core) and Haswell (32 core)".format(m))
	plt.legend(loc='lower right')
	plt.xscale('log')
	plt.xlabel("Number of Points (N)")
	plt.ylabel("Speedup (Compared to Serial)")
	# plt.show()
	plt.savefig('speedup.png')


def upc_strong(n, m, k):
	df = pd.read_csv('upc_hsw.csv')
	df = df[df['n'] == n]

	for i, mi in enumerate(m):
		for j, kj in enumerate(k):
			d = df[(df['m'] == mi) & (df['k'] == kj)]
			c = rancolor()
			print(d)
			plt.scatter(d['p'], d['t'], color=c, label=None)
			plt.plot(d['p'], d['t'], color=c, label='m = {}, k = {}'.format(mi, kj))
	plt.legend()
	plt.title("UPC Strong Scaling on Haswell with N={}".format(int(n)))
	plt.ylabel("Wall time(s)")
	plt.xlabel("Num threads")
	# plt.show()
	plt.savefig('upc{}_6.png'.format(int(n)))


def upc_weak(m, p=128):
	basecolor = 'green'
	colors = ['', 'dark', 'light']
	for i, k in enumerate([2, 5, 10]):
		df = pd.read_csv('upc_hsw.csv')

		df = df[(df['m'] == m) & (df['k'] == k)]
		n = [1e3, 1e4, 1e5, 1e6]
		# print(df)

		c = colors[i] + basecolor
		print(df[df['p'] == 1])
		print(df[df['p'] == 128])
		# speedup = df[df['p'] == 128]['t'].as_matrix()/df[df['p'] == 1].as_matrix()
	# 	plt.plot(n[:len(d['t'])], speedup, color=c, label=lab + ' k={}'.format(k))
	# 	plt.scatter(n[:len(d['t'])], speedup, color=c)
	#
	# plt.title("Speedup vs. N, M={} for KNL (68 core) and Haswell (32 core)".format(m))
	# plt.legend(loc='lower right')
	# plt.xscale('log')
	# plt.xlabel("Number of Points (N)")
	# plt.ylabel("Speedup (Compared to Serial)")
	# # plt.show()
	# plt.savefig('speedup.png')


if __name__ == "__main__":
	# X = generate_data()
	# print("X is: \n", X)
	# C = kpp_serial(X, K)
	#
	# print("C is \n", C)
	# print(C)
	# compare_kmeans()
	# plot_kmeans_pp(s=13)
	# plot_kmeans()
	# plot_kmeans(s=13)
	# omp_strong(comp='hsw', n=1e5, m=[5000, 1000, 10, 1], k=[5], alt_title="Effect of Dimensionality")
	# omp_weak(m=5000, p=16)
	# upc_strong(1e4, [5000], [5, 10])
	upc_weak(m=1000, p=64)
