NVCC = nvcc
CC = g++
CFLAGS = -O3 -std=c++11
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37 -std=c++11
TARGETS = serial

all: $(TARGETS)

serial: serial GPU

serial: serial.o
	$(CC) -o $@ $(CLIBS) serial.o
	rm -f serial.o

serial.o: serial.cpp
	$(CC) -c $(CFLAGS) serial.cpp

gpu: gpu.o
	$(NVCC) -o $@ $(NVCCLIBS) gpu.o
	rm -f gpu.o

gpu.o: gpu.cu
	$(NVCC) -c $(NVCCFLAGS) gpu.cu

clean:
	rm -f $(TARGETS)
