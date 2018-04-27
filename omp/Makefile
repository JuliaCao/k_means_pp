CC = CC
OPENMP = -openmp
#Note: this is the flag for Intel compilers. Change this to -fopenmp for GNU compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
CFLAGS = -O3  -std=c++11
LIBS =
TARGETS = omp omp_alt

all: omp omt_alt
omp: omp.o
	$(CC) -o $@ $(LIBS) $(OPENMP) omp.o
omp.o: omp.cpp
	$(CC) -c $(OPENMP) $(CFLAGS) omp.cpp
omp_alt: omp_alt.o
	$(CC) -o $@ $(LIBS) $(OPENMP) omp_alt.o
omp.o: omp_alt.cpp
	$(CC) -c $(OPENMP) $(CFLAGS) omp_alt.cpp

clean:
	rm -f $(TARGETS) *.o
