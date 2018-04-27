
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
// #include <Eigen/Dense>
// #else
// #include <eigen3/Eigen/Dense>
// #endif
#include </global/homes/a/ardunn/eigen/Eigen/Dense>

using namespace std;
using namespace Eigen;

int N;
int M;
int K;

// template<typename Rand>
//  void generate_data(MatrixXd& data, Rand& r){
//  // MatrixXf data(N,M);

//  for(int i = 0;i<N;i++){
//      for(int j = 0; j < M ; j++){
//          data(i,j) = r();
//      }
//  }

//  return;
// }

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
      s += W(i);
    }

    return i;
}

template<typename Rand>
double kpp_serial(MatrixXd& X, MatrixXd& C, Rand& r) {
    double STime = read_timer( );

    VectorXd D(N);
    for(int i  = 0 ; i < N ; i++){
        D(i) = numeric_limits<float>::max();
    }

    // The first seed is selected uniformly at random
    int index = (int)(r() * N);
    C.row(0) = X.row(index);
    // cout << "picking idx " << index << endl;

    for(int j = 1; j < K; j++){
      for(auto i = 0;i<N;i++){
            VectorXd c = C.row(j-1);
            VectorXd x = X.row(i);
            VectorXd tmp = c - x;
            D(i) = min(tmp.norm(),D(i));
        }

      int i = weighted_rand_index(D,r);
    // cout << "i = " << i << endl;
      C.row(j) = X.row(i);
    }
    // cout << "C =" << C << endl;

    double CTime = read_timer( ) - STime;
    return CTime;
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
    N = read_int( argc, argv, "-n", 100 );
    K = read_int( argc, argv, "-k", 3 );
    M = read_int( argc, argv, "-m", 2 );

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
    double CTime = kpp_serial(X, C, weight_rand);

    if(fsave) {
        fprintf( fsave, "N=%d M=%d K=%d Time=%.2f\n", N, M, K, CTime);
        fclose( fsave );
    }
    // output_kmeans_pp()
}
