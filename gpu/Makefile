CC = nvcc
CFLAGS = -O3 -arch=compute_37 -code=sm_37
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37

all: gpu

gpu: gpu.o common.o
        $(CC) -o $@ $(NVCCLIBS) gpu.o

gpu.o: gpu.cu common.h
        $(CC) -c $(NVCCFLAGS) gpu.cu

clean:
				rm -f *.o $(target)
