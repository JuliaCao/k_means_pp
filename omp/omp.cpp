#include <cstdio>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <omp.h>
#include <iostream>
#include <ctime>
#include <vector>

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif

using namespace std;
using namespace Eigen;

#define M 3
#define K 5
#define N 100


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
int weighted_rand_index_bound(VectorXd& W, Rand& r, int lo, int hi){
    double culmulative = 0;
    for(int i = lo; i < hi; i++){
        culmulative += W(i);
    }

    culmulative *= r();

    int index = lo;
    double s = W(lo);

    while(s < culmulative){
        index++;
        s += W(index);
    }
    return index;
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

		    for(int i = lo;i<hi; i++){
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


int main( int argc, char** argv ){
	srand((unsigned int)time(0));	

	random_device rd;
	// std::mt19937 e2(rd());
	uniform_real_distribution<double> dist(-1.f, 1.f);
	uniform_real_distribution<double> zero_one(0.f, 1.f);
	//auto mat_rand = bind(dist,ref(rd));
	auto weight_rand = bind(zero_one,ref(rd));

	MatrixXd X = MatrixXd::Random(N,M);
	MatrixXd C(K,M);

	cout << "X" << X << endl;	

	// generate_data(X,mat_rand);
  	kpp_openmp(X, C, weight_rand);
	// output_kmeans_pp()
}
