PTXXX=ptxas
FBXX=fatbinary
CXX=g++-7
C=gcc-7
NVXX=nvcc

ARCH=sm_75
PTXFLAGS=-arch=$(ARCH)
FBXXFLAGS=-64

ROOT=..
SRC_DIR=$(ROOT)/src
BUILD_DIR=$(ROOT)/build

CUDALIB=/usr/local/cuda-10.2/lib64
CUDAINCLUDE=/usr/local/cuda-10.2/include

SRCFILES=$(wildcard $(SRC_DIR)/Layers/*.cu)
OBJFILES=$(patsubst $(SRC_DIR)/Layers/%.cu,$(BUILD_DIR)/Layers/%.o,$(SRCFILES))

all: $(BUILD_DIR)/Layers/layers.o $(OBJFILES)
	$(CXX) $(SRC_DIR)/main.cpp $(BUILD_DIR)/Layers/layers.o $(OBJFILES) $(BUILD_DIR)/Tile/mmult.o -O3 -L$(CUDALIB) -lcudart -lcudadevrt -o $(BUILD_DIR)/main -I$(CUDAINCLUDE)	

$(BUILD_DIR)/Layers/layers.o: $(OBJFILES) $(BUILD_DIR)/Tile/mmult.o
	$(NVXX) --gpu-architecture=$(ARCH) --device-link $(BUILD_DIR)/Tile/mmult.o $(OBJFILES) -o $@

$(BUILD_DIR)/Layers/%.o: $(SRC_DIR)/Layers/%.cu
	$(NVXX) -Xptxas="-v -O3" -I/home/jmracek/NVIDIA_CUDA-10.2_Samples/common/inc --gpu-architecture=$(ARCH) -o $@ --device-c $<

$(BUILD_DIR)/Tile/mmult.o: $(SRC_DIR)/Tile/mmult.cu build
	$(NVXX) -Xptxas="-v -O3" --gpu-architecture=$(ARCH) -o $@ --device-c $<

build:
	mkdir -p $(ROOT)/$@ && mkdir -p $(ROOT)/$@/Layers && mkdir -p $(ROOT)/$@/Tile

clean:
	rm -rf ../build
