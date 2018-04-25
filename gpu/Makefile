NVCC = nvcc
CC = g++ -std=c++11
CFLAGS = -O3 -arch=compute_37 -code=sm_37
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37 -std=c++11
TARGETS = gpu serial

all: gpu

gpu: gpu.o
	$(NVCC) -o $@ $(NVCCLIBS) gpu.o
	rm -f gpu.o

serial: serial.o
	$(CC) -o $@ $(CFLAGS) serial.o
	rm -f serial.o

gpu.o: gpu.cu
	$(NVCC) -c $(NVCCFLAGS) gpu.cu

serial.o: serial.cu
	$(CC) -c $(CFLAGS) serial.cu

clean:
	rm -f $(TARGETS)
