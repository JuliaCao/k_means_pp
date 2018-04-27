CC = CC
target = omp
CFLAGS = -O3 -std=c++11
OPENMP = -openmp

all: omp

serial: omp.cpp
	$(CC) $(OPENMP) omp.cpp  -o  omp

clean:
	rm -f $(target)
