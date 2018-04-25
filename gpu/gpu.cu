#include <cstdio>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif

using namespace std;
using namespace Eigen;

#define M 2  // num dimensions
#define K 10  // num clusters
#define N 1000  //num points

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

	std::string sep = "\n----------------------------------------\n";
	int n = read_int( argc, argv, "-n", 1000);
	int m = read_int( argc, argv, "-m", 2);
	int k = read_int( argc, argv, "-k", 10);

	cout << sep << "RUNNING KMEANS++ GPU WITH " << n << " POINTS , " << k << " CLUSTERS, AND " << m << " DIMENSIONS.\n";

	random_device rd;
	// std::mt19937 e2(rd());
	uniform_real_distribution<double> kmdata(-1.f, 1.f);
	uniform_real_distribution<double> zero_one(0.f, 1.f);
	auto weight_rand = bind(zero_one, ref(rd));

	MatrixXd X = MatrixXd::Random(n, m);
	//cout << X << sep;
	MatrixXd C(k, m);

	// auto mat_rand = bind(kmdata, ref(rd));
	// generate_data(X,mat_rand);
	double t0 = read_timer( );
  kpp_gpu(n, k, X, C, weight_rand);
	double t1 = read_timer( ) - t0;

	cout << "THE GPU SIMULATION TOOK " << t1 << " SECONDS." << sep;
	//cout << C << sep;
	// output_kmeans_pp()
}
