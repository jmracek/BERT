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
	$(CXX) $(SRC_DIR)/main.cpp $(BUILD_DIR)/Layers/layers.o $(OBJFILES) -O3 -L$(CUDALIB) -lcudart -o $(BUILD_DIR)/main -I$(CUDAINCLUDE)	

$(BUILD_DIR)/Layers/layers.o: $(OBJFILES)
	$(NVXX) --gpu-architecture=$(ARCH) --device-link $(OBJFILES) -o $@

$(BUILD_DIR)/Layers/%.o: $(SRC_DIR)/Layers/%.cu build
	$(NVXX) -Xptxas="-v -O3" --gpu-architecture=$(ARCH) -o $@ --device-c $<

build:
	mkdir -p $(ROOT)/$@ && mkdir -p $(ROOT)/$@/Layers

clean:
	rm -rf ../build
