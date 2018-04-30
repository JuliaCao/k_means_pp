
#include <cstdio>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <iostream>
#include <ctime>
#include <upcxx/upcxx.hpp>
#include <time.h>
#include <sys/time.h>

// #if defined __GNUC__ || defined __APPLE__
// #include <Eigen/Dense>
// #else
// #include <eigen3/Eigen/Dense>
// #endif
#include </global/homes/j/juliacao/eigen/Eigen/Dense>

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

struct GmatrixXd
{
    upcxx::global_ptr<int64_t> I;
    upcxx::global_ptr<double> S;
    std::vector<upcxx::global_ptr<double>> X; //has length #threads
    // std::vector<upcxx::global_ptr<VectorXd>> C;
    upcxx::global_ptr<double> C;

    size_t my_Xsize;
    size_t my_Csize;

    size_t blk;
    size_t blk_size;

    size_t size() const noexcept;


    GmatrixXd(size_t Xsize, size_t Csize){
        // S.reXsize(upcxx::rank_n());
        my_Xsize = Xsize;
        my_Csize = Csize * M;

        X.resize(upcxx::rank_n());

        blk = (Xsize+upcxx::rank_n()-1)/upcxx::rank_n();
        blk_size = blk * M;

        upcxx::global_ptr<double> local_data;
        local_data = upcxx::new_array<double>(blk_size);

        VectorXd v1;
        for (int i = 0; i < blk; i++) {
          v1 = VectorXd::Random(M);
          vector<double> v2(v1.data(), v1.data() + v1.size());

          // cout << "X" << i << endl;
          for (int j = 0; j < M; j++) {
            // cout << v2[j] << " " ;
            upcxx::rput(v2[j], local_data+i*M+j).wait();
          }
          // cout << " " << endl;
        }

        // for (int i = 0; i < blk_size; i++) {
        //   double tmp = upcxx::rget(local_data+i).wait();
        //   cout << tmp << endl;
        //   if (i % 3 == 2) {
        //     cout << " " << endl;
        //   }
        // }

        X[upcxx::rank_me()] = local_data;

        for (int i = 0; i < upcxx::rank_n(); ++i)
        {
            X[i] = upcxx::broadcast(X[i], i).wait();
        }

        if (upcxx::rank_me() == 0) {
            I = upcxx::new_array<int64_t>(upcxx::rank_n());
            S = upcxx::new_array<double>(upcxx::rank_n());
            C = upcxx::new_array<double>(my_Csize);  
        }
        I = upcxx::broadcast(I, 0).wait();
        S = upcxx::broadcast(S, 0).wait();
        C = upcxx::broadcast(C, 0).wait();

    }

    void write_C_slot(const int64_t slot, const vector<double>& v2) {
      static int x=0;
      // cout << "write_C_slot " << x << endl;
      for (int i = 0; i < M; i++) {
        // cout << v2[i] << endl;
        upcxx::rput(v2[i],C+slot*M+i).wait();
        // cout << "write index" << slot*M+i << endl;
      }
      // cout << "written " << x++ << endl;
    }

    void write_X_to_C(const int64_t X_slot,const int64_t C_slot){
        int rank = X_slot / blk;
        int pos = X_slot % blk;

        vector<double> vec_holder;
        vec_holder.resize(M);

        for (int i = 0; i < M; i++) {
          double tmp = upcxx::rget(X[rank]+pos*M+i).wait();
          vec_holder[i] = tmp;
        }

        write_C_slot(C_slot,vec_holder);
    }

    int64_t get_I_slot(const int64_t slot) {
      return upcxx::rget(I+slot).wait();
    }

    void write_S_slot(const int64_t slot,const double S_i){
        upcxx::rput(S_i,S+slot).wait();
    }

  //   void printX_vector_i(const int64_t slot){
  //     int rank = slot / blk;
  //     int pos = slot % blk;
  //     VectorXd v = upcxx::rget(X[rank]+pos).wait();
  //     cout << "getting slot" << slot << "\n with value \n" <<v << endl;
  // }

    VectorXd get_C_row(const int64_t slot) {
      vector<double> vec_holder;
      vec_holder.resize(M);

      for (int i = 0; i < M; i++) {
        // cout << "read index" << slot*M+i << endl;
        double tmp = upcxx::rget(C+slot*M+i).wait();
        vec_holder[i] = tmp;
        // cout << tmp << endl;
      }

      VectorXd v2 = Map<VectorXd, Unaligned>(vec_holder.data(), vec_holder.size());
      return v2;
    }

    VectorXd get_X_row(const int64_t rank, const int64_t pos) {
        vector<double> vec_holder;
        vec_holder.resize(M);

        for (int i = 0; i < M; i++) {
          double tmp = upcxx::rget(X[rank]+pos*M+i).wait();
          vec_holder[i] = tmp;
        }

        VectorXd v2 = Map<VectorXd, Unaligned>(vec_holder.data(), vec_holder.size());
        return v2;
    }
};

template<typename Rand>
double kpp_upc(GmatrixXd& XX, Rand& r){
    double STime = read_timer( );

    int p = upcxx::rank_n();//#threads
	   cout << "#p: " << p << endl;
    // vector<int> I(p,0);
    double S;
    // VectorXd S(p);
    // vector<double> D(N,0);
    VectorXd D(XX.blk);

    for(int i  = 0 ; i < XX.blk ; i++){
        D(i) = numeric_limits<float>::max();
    }

    // The first seed is selected uniformly at random
    int index = (int)(r() * N);

    // cout << "index" << index << endl;

    if(upcxx::rank_me() == 0){
        XX.write_X_to_C(index,0);   
    }

    for(int j = 1; j < K; j++) {
        int t = upcxx::rank_me();
        // int lo = t * (N / p);
        // int hi = min(lo + N/p, N-1);

        //calculate weights for this part

        S = 0.0f;

        // for(auto i = lo;i<hi; i++){
        for(int i = 0; i < XX.blk ; i++) {
            VectorXd C_row = XX.get_C_row(j-1);

      	    // int rank = i / XX.blk;
       	    // int pos = i % XX.blk;
 
            VectorXd X_row = XX.get_X_row(t, i);

            if(j == 1){
                D(i) = (C_row - X_row).norm();
            }
            else{
                D(i) = min((X_row - C_row).norm(),D(i));
            }
            S = S + D(i);
        }

        // cout << "exited for loop" << endl;

        int64_t sub_i = weighted_rand_index_bound(D,r,0,XX.blk);
        // cout << "sub_i " << endl;
        // int sub_i = weighted_rand_index(D,r);
       
        // I[t] = sub_i;
        upcxx::rput(sub_i,XX.I+t).wait();
        // cout << "rput" << endl;
        // upcxx::rput(S,XX.S[upcxx::rank_me()]).wait();
        XX.write_S_slot(upcxx::rank_me(), S);
        // cout << "finished write_S_slot" << endl;
  
        upcxx::barrier();

        if(upcxx::rank_me() == 0) {
          // change S
          VectorXd S(upcxx::rank_n());
          for (int i = 0; i < upcxx::rank_n(); ++i)
          {
              double tmp = upcxx::rget(XX.S+i).wait();
              S(i) = tmp;
          }
          int sub_t = weighted_rand_index(S,r);
          // int i = I[sub_t];
          int i = XX.get_I_slot(sub_t);

          // cout << "selected " << i << endl;
      
          XX.write_X_to_C(i,j);

        }

        upcxx::barrier();
      // C.row(j) = X.row(i);
    }

    double CTime = read_timer( ) - STime;
    return CTime;

    // cout << "final C" << endl;
    // for (int i = 0; i < K; i++) {
    //   VectorXd C_row = XX.get_C_row(i);
    //   cout << C_row << endl;
    // }
}


int main( int argc, char** argv ){
    srand((unsigned int)time(0));

    N = read_int( argc, argv, "-n", 100 );
    K = read_int( argc, argv, "-k", 3 );
    M = read_int( argc, argv, "-m", 2 );

    char *savename = read_string( argc, argv, "-o", NULL );
    FILE *fsave = savename ? fopen( savename, "a" ) : NULL;

    random_device rd;
    // std::mt19937 e2(rd());
    uniform_real_distribution<double> dist(-1.f, 1.f);
    uniform_real_distribution<double> zero_one(0.f, 1.f);
    //auto mat_rand = bind(dist,ref(rd));
    auto weight_rand = bind(zero_one,ref(rd));

    upcxx::init();
    GmatrixXd XX(N,K);

    kpp_upc(XX, weight_rand);

    double CTime = kpp_upc(XX, weight_rand);

    int p = upcxx::rank_n();//#threads

    if (upcxx::rank_me() == 0) {
      if(fsave) {
          fprintf( fsave, "N=%d M=%d K=%d Rank=%d Time=%.2f\n", N, M, K, p, CTime);
          fclose( fsave );
      }
    }

    upcxx::finalize();

    // output_kmeans_pp()
}
