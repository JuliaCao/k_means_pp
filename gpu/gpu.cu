#include <cstdio>
#include <cstdlib>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

/*
Author: Alexander Dunn
University of California, 2018
CS 267 Final Project: Parallelizing K-means++ Initialization
GPU portion
*/

using namespace std;

// Command line parsing and timing: from Homework 2.3 starter files common.cu
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

float randarr(float low, float high, int m) {
		ranarr float[m];
		for (int j = 1; j < m; j++){
			ranarr[i] = low + ((float) rand()) / (float) RAND_MAX * (hi - low));
		}
		return ranarr;
}
//////////////////////////////////////////////////////////////////////////////

// GPU Indexing

// template<typename Rand>
// struct prob_reduce
// {
//     __host__ __device__
//         tuple<float, int> operator()(const tuple<float, int>& t1, const tuple<float, int>& t2) const {
// 					float w1 = get<0>(t1);
// 					float w2 = get<0>(t2);
// 					int i1 = get<1>(t1);
// 					int i2 = get<1>(t2);
// 					float rval = 0.2837472 * (w1 + w2);
// 					if (rval > w1){
// 						return make_tuple(w1 + w2, i2);
// 					}
// 					else{
// 						return make_tuple(w1 + w2, i1);
// 					}
//         }
// };

// struct D_functor
// {
//     const VectorXd c;
//     D_functor(VectorXd _c) : c(_c) {}
//
//     __host__ __device__
//         float operator()(const VectorXd& x, const float& d) const {
// 					VectorXd d2 = x - c;
//           return min(d2.norm(), d);
//         }
// };

// // GPU Algorithm
// template<typename Rand>
// void kpp_gpu(int n, int k, thrust::device_vector<float> &D,
// 	thrust::device_vector<int> I, thrust::device_vector<VectorXd> &X,
// 	thrust::device_vector<VectorXd> &C, Rand &r) {
//
// 	// The first seed is selected uniformly at random
// 	int index = (int)(r() * n);
// 	C[0] = X[index];
// 	for(int j = 1; j < k; j++){
// 			// thrust::transform(X.begin(), X.end(), D.begin(), D.begin(), D_functor(C[j-1]));
// 			// thrust::reduce(D.begin(), D.end(), I.begin(), I.end(), )
// 			// C[j] = X[i];
// 			}
// 	return;
// }

int main( int argc, char** argv ){

	// Parsing commands and prettify-ing output
	std::string sep = "\n-----------------------------------------------------------------------------------------------------------\n";
	int n = read_int( argc, argv, "-n", 1000);
	int m = read_int( argc, argv, "-m", 2);
	int k = read_int( argc, argv, "-k", 10);

	// Initializing Data
	random_device rd;
	uniform_real_distribution<double> zero_one(0.f, 1.f);
	auto weight_rand_gpu = bind(zero_one, ref(rd));
	float inf = numeric_limits<float>::max();
	// thrust::device_vector<array> C(k);
	// thrust::device_vector<array> X(n);
	// thrust::device_vector<float> D(n);
	// thrust::fill(D.begin(), D.end(), inf);
	// thrust::device_vector<int> I(n);
	// thrust::sequence(I.begin(), I.end());

	// Populating gpu arrays
	for (int i  = 0 ; i < n ; i++){
		X[i] = randarr(m, 2.0, 3.0);
	}

	cout << X;

	// // Running GPU simulation
	// cout << sep << "RUNNING KMEANS++ GPU WITH " << n << " POINTS , " << k << " CLUSTERS, AND " << m << " DIMENSIONS.\n";
	// double t0 = read_timer( );
  // kpp_gpu(n, k, D_gpu, I_gpu, X_gpu, C_gpu, weight_rand_gpu);
	// double t1 = read_timer( ) - t0;
	// cout << "THE GPU SIMULATION TOOK " << t1 << " SECONDS. \n";
}
