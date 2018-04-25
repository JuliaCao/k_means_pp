#include <cstdio>
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

/*
Author: Alexander Dunn
University of California, 2018
CS 267 Final Project: Parallelizing K-means++ Initialization
GPU portion
*/

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif

using namespace std;
using namespace Eigen;

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


// k-means++
template<typename Rand>
int weighted_rand_index(VectorXd& W, Rand& r){
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
void kpp_gpu(int n, int k, MatrixXd &X, MatrixXd &C, Rand &r) {

	float inf = numeric_limits<float>::max();
	thrust::device_vector<float> D_gpu(n);
	thrust::fill(D_gpu.begin(), D_gpu.end(), inf);
	thrust::device_ptr<float> D_gpu_ptr = D_gpu.data()
	//float* D_gpu_ptr = &D_gpu[0];
	Map<VectorXd> D(D_gpu_ptr, n);

	// The first seed is selected uniformly at random
	int index = (int)(r() * n);
	C.row(0) = X.row(index);
	for(int j = 1; j < k; j++){
			for(auto i = 0; i < n;i++){
					VectorXd c = C.row(j-1);
					VectorXd x = X.row(i);
					VectorXd tmp = c - x;
				D(i) = min(tmp.norm(), D(i));
			}

		int i = weighted_rand_index(D,r);
	C.row(j) = X.row(i);
	}
	return;
}

template<typename Rand>
void kpp_serial(int n, int k, MatrixXd &X, MatrixXd &C, Rand &r) {

	VectorXd D(n);
	for(int i  = 0 ; i < n ; i++){
		D(i) = numeric_limits<float>::max();
	}

	// The first seed is selected uniformly at random
	int index = (int)(r() * n);
	C.row(0) = X.row(index);
	for(int j = 1; j < k; j++){
			for(auto i = 0; i < n;i++){
					VectorXd c = C.row(j-1);
					VectorXd x = X.row(i);
					VectorXd tmp = c - x;
				D(i) = min(tmp.norm(), D(i));
			}

		int i = weighted_rand_index(D,r);
	C.row(j) = X.row(i);
	}
	return;
}


int main( int argc, char** argv ){

	// Parsing commands and prettify-ing output
	std::string sep = "\n-----------------------------------------------------------------------------------------------------------\n";
	int n = read_int( argc, argv, "-n", 1000);
	int m = read_int( argc, argv, "-m", 2);
	int k = read_int( argc, argv, "-k", 10);

	// Initializing Data
	cout << sep << "RUNNING KMEANS++ GPU WITH " << n << " POINTS , " << k << " CLUSTERS, AND " << m << " DIMENSIONS.\n";
	random_device rd;
	// std::mt19937 e2(rd());
	uniform_real_distribution<double> zero_one(0.f, 1.f);
	auto weight_rand = bind(zero_one, ref(rd));
	MatrixXd X = MatrixXd::Random(n, m);
	MatrixXd C(k, m);
	// Running GPU simulation
	double t0 = read_timer( );
  kpp_gpu(n, k, X, C, weight_rand);
	double t1 = read_timer( ) - t0;
	cout << "THE GPU SIMULATION TOOK " << t1 << " SECONDS. \n";


	// Initializing Data
	cout << "RUNNING KMEANS++ SERIAL WITH SAME " << n << " POINTS , " << k << " CLUSTERS, AND " << m << " DIMENSIONS.\n";
	C = MatrixXd::Zero(k, m);
	// Running serial simulation
	double t2 = read_timer( );
  kpp_serial(n, k, X, C, weight_rand);
	double t3 = read_timer( ) - t2;
	cout << "THE SERIAL/CPU SIMULATION TOOK " << t1 << " SECONDS. \n";
	cout << "THE RESULTING SPEEDUP IS: " << t3/t1 << sep;
}
