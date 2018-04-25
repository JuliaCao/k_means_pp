CC = gcc -std=gcc++0x
target = opm
OPENMP = -fopenmp

all: omp

serial: omp.cpp
	$(CC) $(OPENMP) serial.cpp  -o  serial

clean:
	rm -f $(target)
