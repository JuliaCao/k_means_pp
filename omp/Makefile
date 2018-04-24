CC = g++ -std=c++11
LDLIBS = -openmp
target = omp

all: omp

omp: omp.cpp
	$(CC) omp.cpp -o serial $(LDLIBS)

clean:
	rm -f $(target)
