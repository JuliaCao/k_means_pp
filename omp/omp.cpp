#include <cstdio>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <omp.h>
#include <iostream>
#include <ctime>
#include <time.h>
#include <sys/time.h>

// #if defined __GNUC__ || defined __APPLE__
#include </global/homes/a/ardunn/eigen/Eigen/Dense>
// #else
// #include <eigen3/Eigen/Dense>
// #endif

using namespace std;
using namespace Eigen;

int N;
int M;
int K;

int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
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

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

template<typename Rand>
int weighted_rand_index(VectorXd& W,Rand& r){
	double culmulative = W.sum() * r();
	int i = 0;
	double s = W(0);
	while (s < culmulative){
		i++;
	  s += abs(W(i));
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
        s += abs(W(index));
    }
    return index;
}


template<typename Rand>
double kpp_openmp(MatrixXd& X,MatrixXd& C, Rand& r){
    int p;
    #pragma omp parallel
    {
    p = omp_get_num_threads();//#threads
    //cout << "total # of threads" << p << endl;
    }

    vector<int> I(p,0);
    VectorXd S(p);

    VectorXd D(N);

    double STime = read_timer( );

    #pragma omp parallel for
    for(int i  = 0 ; i < N ; i++){
    	D(i) = numeric_limits<float>::max();
    }

    // The first seed is selected uniformly at random
    int index = (int)r() * N;
    C.row(0) = X.row(index);

    for(int j = 1; j < K; j++){

    	#pragma omp parallel for schedule(dynamic)
    	for(int t = 0; t < p; t++){
	//	cout << "thread" << omp_get_thread_num() << endl;
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

		    int sub_i = weighted_rand_index_bound(D,r,lo,hi);
		    //cout << "select idx per thread" << sub_i << endl;
		    I[t] = sub_i;

    	}

    	// for(auto i = 0;i<N;i++){
    	// 	D(i) = min((X(i) - C(j-1)).norm(),D(i));
    	// }

      int sub_t = weighted_rand_index(S,r);
      int i = I[sub_t];
//      cout << "select idx" << i << endl;
      C.row(j) = X.row(i);
//	cout << "C" << C << endl;
    }
    double CTime = read_timer( ) - STime;
    return CTime;
}


int main( int argc, char** argv ){
    N = read_int( argc, argv, "-n", 100 );
    K = read_int( argc, argv, "-k", 3 );
    M = read_int( argc, argv, "-m", 2 );
    int P = read_int( argc, argv, "-p", 1 );

		omp_set_num_threads(P);

    char *savename = read_string( argc, argv, "-o", NULL );
    FILE *fsave = savename ? fopen( savename, "a" ) : NULL;

    srand((unsigned int)time(0));

    random_device rd;
    // std::mt19937 e2(rd());
    uniform_real_distribution<double> dist(-1.f, 1.f);
    uniform_real_distribution<double> zero_one(0.f, 1.f);
    //auto mat_rand = bind(dist,ref(rd));
    auto weight_rand = bind(zero_one,ref(rd));

    MatrixXd X = MatrixXd::Random(N,M);
    MatrixXd C(K,M);

    // cout << "X" << X << endl;

    // generate_data(X,mat_rand);
    double CTime = kpp_openmp(X, C, weight_rand);

    if(fsave) {
        fprintf( fsave, "N=%d M=%d K=%d Time=%.2f #threads=%d\n",
        	N, M, K, CTime, P);
        fclose( fsave );
    }
    // output_kmeans_pp()
}
