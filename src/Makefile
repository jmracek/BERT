PTXXX=ptxas
FBXX=fatbinary
CXX=g++-7
C=gcc-7
NVXX=nvcc

ARCH=sm_75
PTXFLAGS=-arch=$(ARCH)
FBXXFLAGS=-64

CUDALIB=/usr/local/cuda-10.2/lib64
CUDAINCLUDE=/usr/local/cuda-10.2/include

all: link
	$(CXX) main.cpp mmult.o link.o -O3 -L$(CUDALIB) -lcudart -o main -I$(CUDAINCLUDE)

link: device
	$(NVXX) --gpu-architecture=$(ARCH) --device-link mmult.o -o link.o

device:
	$(NVXX) -Xptxas="-v -O3" --gpu-architecture=$(ARCH) --device-c mmult.cu


#preprocess: mmamult.ptx
#	$(C) -E - < $< > mmamult_proc.ptx




clean:
	rm mmult.o mmamult.o main link.o
