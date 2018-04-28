
#include <cstdio>
#include <random>
#include <limits>
#include <functional>
#include <cstdint>
#include <omp.h>
#include <iostream>
#include <ctime>
#include <upcxx/upcxx.hpp>

// #if defined __GNUC__ || defined __APPLE__
// #include <Eigen/Dense>
// #else
// #include <eigen3/Eigen/Dense>
// #endif
#include </global/homes/j/juliacao/eigen/Eigen/Dense>

using namespace std;
using namespace Eigen;

#define M 3
#define K 2
#define N 10




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
void kpp_serial(MatrixXd& X, MatrixXd& C, Rand& r) {

    VectorXd D(N);
    for(int i  = 0 ; i < N ; i++){
        D(i) = numeric_limits<float>::max();
    }

    // The first seed is selected uniformly at random
    int index = (int)(r() * N);
    C.row(0) = X.row(index);
    cout << "picking idx " << index << endl;

    for(int j = 1; j < K; j++){
      for(auto i = 0;i<N;i++){
            VectorXd c = C.row(j-1);
            VectorXd x = X.row(i);
            VectorXd tmp = c - x;
            D(i) = min(tmp.norm(),D(i));
        }

      int i = weighted_rand_index(D,r);
    cout << "i = " << i << endl; 
      C.row(j) = X.row(i);
    }
    cout << "C =" << C << endl;

    return;
}

struct GmatrixXd
{
    upcxx::global_ptr<int64_t> I;
    upcxx::global_ptr<double> S;
    std::vector<upcxx::global_ptr<VectorXd>> X; //has length #threads
    // std::vector<upcxx::global_ptr<VectorXd>> C;
    upcxx::global_ptr<VectorXd> C;

    size_t my_Xsize;
    size_t my_Csize;

    size_t blk;

    size_t size() const noexcept;


    GmatrixXd(size_t Xsize, size_t Csize){
        // S.reXsize(upcxx::rank_n());
        my_Xsize = Xsize;
        my_Csize = Csize;

        X.resize(upcxx::rank_n());

        blk = (Xsize+upcxx::rank_n()-1)/upcxx::rank_n();

        upcxx::global_ptr<VectorXd> local_data;
        local_data = upcxx::new_array<VectorXd>(blk);

        for (int i = 0; i < blk; i++) {
          VectorXd tmp = VectorXd::Random(M);
          upcxx::rput(tmp, local_data+i).wait();
          cout << "X" << i << ": " << tmp << endl;
        }

        X[upcxx::rank_me()] = local_data;

        for (int i = 0; i < upcxx::rank_n(); ++i)
        {
            X[i] = upcxx::broadcast(X[i], i).wait();
        }

        if (upcxx::rank_me() == 0) {
            I = upcxx::new_array<int64_t>(upcxx::rank_n());
            S = upcxx::new_array<double>(upcxx::rank_n());
            C = upcxx::new_array<VectorXd>(Csize);  
        }
        I = upcxx::broadcast(I, 0).wait();
        S = upcxx::broadcast(S, 0).wait();
        C = upcxx::broadcast(C, 0).wait();

    }

    void write_C_slot(const int64_t slot,const VectorXd& C_i){
        upcxx::rput(C_i,C+slot).wait();
    }

    void write_X_to_C(const int64_t X_slot,const int64_t C_slot){
        int rank = X_slot / blk;
        int pos = X_slot % blk;
	VectorXd selected =  upcxx::rget(X[rank]+pos).wait();
        write_C_slot(C_slot,selected);
    }

    int64_t get_I_slot(const int64_t slot) {
      return upcxx::rget(I+slot).wait();
    }

    void write_S_slot(const int64_t slot,const double S_i){
        upcxx::rput(S_i,S+slot).wait();
    }

};

template<typename Rand>
void kpp_upc(GmatrixXd& XX, Rand& r){

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

    cout << "got here1" << endl;

    // The first seed is selected uniformly at random
    int index = (int)(r() * N);

    if(upcxx::rank_me() == 0){
        XX.write_X_to_C(index,0);   
    }

	cout << "selected C[0]" << endl;
    for(int j = 1; j < K; j++) {
        int t = upcxx::rank_me();
        int lo = t * (N / p);
        int hi = min(lo + N/p, N-1);

        //calculate weights for this part

        S = 0.0f;

        // for(auto i = lo;i<hi; i++){
        for(int i = 0; i < hi - lo ; i++){
            cout << "in loop" << endl;
            VectorXd C_row = upcxx::rget(XX.C+j-1).wait();
	    cout << "C_row" << C_row << endl;
	    int rank = (i+lo) / XX.blk;
 	    int pos = (i + lo) % XX.blk;
	    cout << "rank " << rank << "pos " << pos << endl;
		cout << XX.X[rank] << endl;
		cout << XX.X[rank]+pos << endl;
            VectorXd X_row = upcxx::rget(XX.X[rank]+pos).wait();
	    cout << "X_row:"<< endl;
	    cout << X_row << endl;

            if(j == 1){
                D(i) = (C_row - X_row).norm();
            }
            else{
                D(i) = min((X_row - C_row).norm(),D(i));
            }
            S = S + D(i);
        }

	cout << "end of loop" << endl;

        int64_t sub_i = weighted_rand_index_bound(D,r,lo,hi);
        // int sub_i = weighted_rand_index(D,r);
       
        // I[t] = sub_i;
        upcxx::rput(sub_i,XX.I+t).wait();
        // upcxx::rput(S,XX.S[upcxx::rank_me()]).wait();
        XX.write_S_slot(upcxx::rank_me(), S);
  
        upcxx::barrier();

	cout << "written to slot" << endl;

        if(upcxx::rank_me() == 0){
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
      
          XX.write_X_to_C(i,j);
        }

        upcxx::barrier();
      // C.row(j) = X.row(i);
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


    // MatrixXd X = MatrixXd::Random(N,M);


    MatrixXd C(K,M);

    // generate_data(X,mat_rand);
    upcxx::init();
    GmatrixXd XX(N,K);

    kpp_upc(XX, weight_rand);

    upcxx::finalize();

    // output_kmeans_pp()
}
