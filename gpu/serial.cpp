#include <cstdio>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <iostream>

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

int main( int argc, char** argv ){

	for (int i = 1; i < argc; ++i) {
		cout << argv[i];
	}

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
