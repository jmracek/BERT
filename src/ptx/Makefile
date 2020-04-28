PTXXX=ptxas
FBXX=fatbinary
CXX=g++
C=gcc
NVXX=nvcc

ARCH=sm_70
PTXFLAGS=-arch=$(ARCH)
FBXXFLAGS=-64

CUDALIB=/usr/local/cuda-10.2/lib64
CUDAINCLUDE=/usr/local/cuda-10.2/include

all: link
	$(CXX) main.cpp mmult.o mmamult.o link.o -O3 -L$(CUDALIB) -lcudart -o main -I$(CUDAINCLUDE)

link: device ptx
	$(NVXX) --gpu-architecture=$(ARCH) --device-link mmult.o mmamult.o -o link.o

device:
	nvcc -Xptxas="-O3" --gpu-architecture=$(ARCH) --device-c mmult.cu

ptx: mmamult.ptx
	$(NVXX) -Xptxas="-O3" --gpu-architecture=$(ARCH) --device-c mmamult.ptx

#preprocess: mmamult.ptx
#	$(C) -E - < $< > mmamult_proc.ptx




clean:
	rm mmult.o mmamult.o main link.o
