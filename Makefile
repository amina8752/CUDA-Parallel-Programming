NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60

all: main

main: main.cu
	$(NVCC) $(NVCCFLAGS) main.cu -o main

clean:
	rm -f main