CC = nvcc
CFLAGS = -O3 -arch=compute_37 -code=sm_37
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37

all: gpu

gpu: gpu.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o

gpu.o: gpu.cu
	$(CC) -c $(NVCCFLAGS) gpu.cu

clean:
	rm -f *.o $(target)
