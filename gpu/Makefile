CC = nvcc
CFLAGS = -O3 -arch=compute_37 -code=sm_37
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37 -std=c++11
target = gpu

all: gpu

gpu: gpu.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o

gpu.o: gpu.cu
	$(CC) -c $(NVCCFLAGS) gpu.cu
	rm -f *.o

clean:
	rm -f $(target)
