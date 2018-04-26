NVCC = nvcc
CC = g++
CFLAGS = -O3 -std=c++11
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37 -std=c++11
TARGETS = serial

all: $(TARGETS)

serial: serial.cpp
	$(CC) serial.cpp -o serial $(CFLAGS)

gpu: gpu.o
	$(NVCC) -o $@ $(NVCCLIBS) gpu.o
	rm -f gpu.o

gpu.o: gpu.cu
	$(NVCC) -c $(NVCCFLAGS) gpu.cu

clean:
	rm -f $(TARGETS)
