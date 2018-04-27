CC = CC
OPENMP = -openmp
#Note: this is the flag for Intel compilers. Change this to -fopenmp for GNU compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
CFLAGS = -O3  -std=c++11
LIBS =

TARGETS = omp

all: omp


omp: omp.o
        $(CC) -o $@ $(LIBS) $(OPENMP) omp.o

omp.o: omp.cpp
        $(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp

clean:
	rm -f $(TARGETS)
