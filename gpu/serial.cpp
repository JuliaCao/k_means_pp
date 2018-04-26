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

/*
Author: Alexander Dunn
University of California, 2018
CS 267 Final Project: Parallelizing K-means++ Initialization
GPU portion, serial comparison file
*/

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


// Serial Indexing
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

// Serial Algorithm
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


	random_device rd;
	uniform_real_distribution<double> dist(-1.f, 1.f);
	uniform_real_distribution<double> zero_one(0.f, 1.f);
	auto weight_rand = bind(zero_one,ref(rd));

	// Initializing Serial Data
	MatrixXd X = MatrixXd::Random(n, m);
	MatrixXd C(k, m);

  kpp_serial(X, C, weight_rand);

	// Initializing Data
	cout << "RUNNING KMEANS++ SERIAL WITH SAME " << n << " POINTS , " << k << " CLUSTERS, AND " << m << " DIMENSIONS.\n";
	// Running serial simulation
	double t2 = read_timer( );
  kpp_serial(n, k, X, C, weight_rand);
	double t3 = read_timer( ) - t2;
	cout << "THE SERIAL/CPU SIMULATION TOOK " << t3 << " SECONDS. \n";
}
