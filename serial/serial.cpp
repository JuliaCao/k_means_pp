#include <cstdio>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <omp.h>

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif

using namespace std;
using namespace Eigen;

#define M 2
#define K 10
#define N 1000

// template<typename Rand>
//  void generate_data(MatrixXd& data, Rand& r){
// 	// MatrixXf data(N,M);

// 	for(int i = 0;i<N;i++){
// 		for(int j = 0; j < M ; j++){
// 			data(i,j) = r();
// 		}
// 	}

// 	return;
// }

template<typename Rand>
int weighted_rand_index(VectorXd& W,Rand& r){
	double culmulative = W.sum() * r();
	int i = 0;
	double s = W(0);
	while (s < culmulative){
		i++;
	  s += W(i);
	}
		
	return i;
}
	
template<typename Rand>
void kpp_serial(MatrixXd& X, MatrixXd& C, Rand& r) {

	VectorXd D(N);
	for(int i  = 0 ; i < N ; i++){
		D(i) = numeric_limits<float>::max();
	}

	// The first seed is selected uniformly at random
	int index = (int)r() * N;
	C(0) = X(index);

	for(int j = 1; j < K; j++){
   	  for(auto i = 0;i<N;i++){
      	VectorXd c = C.row(j-1);
        VectorXd x = X.row(i);
        VectorXd tmp = c - x;
    		D(i) = min(tmp.norm(),D(i));
    	}
	  
	  int i = weighted_rand_index(D,r);
	  C(j) = X(i);
	}
  
	return;
}

template<typename Rand>
void kpp_openmp(MatrixXd& X,MatrixXd& C, Rand& r){
	

    int p = omp_get_num_threads();//#threads
    vector<int> I(p,0);
    VectorXd S(p);

    VectorXd D(N);

    #pragma omp parallel for
    for(int i  = 0 ; i < N ; i++){
    	D(i) = numeric_limits<float>::max();
    }

    // The first seed is selected uniformly at random
    int index = (int)r() * N;
    C(0) = X(index);

    for(int j = 1; j < K; j++){

    	#pragma omp parallel for
    	for(int t = 0; t < p; t++){
    		int lo = t * (N / p);
    		int hi = min(lo + N/p, N-1);

    		//calculate weights for this part
    		

		    S(t) = 0.0f;

		    for(auto i = lo;i<hi; i++){
		    	if(j == 1){
		    		D(i) = (C.row(j-1) - X.row(i)).norm();
		    	}
		    	else{
		    		D(i) = min((X.row(i) - C.row(j-1)).norm(),D(i));
		    	}
		    	S(t) = S(t) + D(i);
		    }

		    int sub_i = lo+ weighted_rand_index_bound(D,r,lo,hi);

		    I[t] = sub_i;

    	}

    	// for(auto i = 0;i<N;i++){
    	// 	D(i) = min((X(i) - C(j-1)).norm(),D(i));
    	// }

      int sub_t = weighted_rand_index(S,r);
      int i = I[sub_t];           
      C(j) = X(i);
    }
  
    return;

}

// def output_kmeans_pp():
//     if X.shape[1] != 2 or C.shape[1] != 2:
//         raise ValueError("M must be 2 dimensions to plot.")
//     plt.scatter(X[:, 0], X[:, 1], color='black')
//     plt.scatter(C[:, 0], C[:, 1], color='red')
//     plt.show()


// def plot_kmeans():
//     if X.shape[1] != 2 or C.shape[1] != 2:
//         raise ValueError("M must be 2 dimensions to plot.")
//     km = KMeans(n_clusters=K, init=C, n_init=1)
//     labels = km.fit_predict(X)
//     rancols = [np.random.rand(3).tolist() for _ in range(max(labels) + 1)]
//     colmap = dict(zip(range(max(labels) + 1), rancols))
//     colors = [colmap[l] for l in labels]
//     for i in range(X.shape[0]):
//         plt.scatter(X[i][0], X[i][1], c=colors[i], s=3)
//     plt.show()

// def compare_kmeans(nruns=1000):
//     initnames = ["ours", "sklearn", "random"]
//     for i, init in enumerate([C, 'k-means++', 'random']):
//         iters = []
//         for run in range(nruns):
//             # print("Run {} in {}".format(run, init))
//             km = KMeans(n_clusters=K, init=init, n_init=1)
//             km.fit_predict(X)
//             iters.append(km.n_iter_)
//         print("Average {} n_iters needed: {}".format(initnames[i], np.mean(iters)))



int main( int argc, char** argv ){

	random_device rd;
	// std::mt19937 e2(rd());
	uniform_real_distribution<double> dist(-1.f, 1.f);
	uniform_real_distribution<double> zero_one(0.f, 1.f);
	auto mat_rand = bind(dist,ref(rd));
	auto weight_rand = bind(zero_one,ref(rd));

	MatrixXd X = MatrixXd::Random(N,M);
	MatrixXd C(K,M);


	// generate_data(X,mat_rand);
  kpp_serial(X, C, weight_rand);
	// output_kmeans_pp()
}